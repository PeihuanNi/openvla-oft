"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
import os
import math
import time
import types
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
try:
    from torch.profiler import ProfilerActivity, profile as torch_profile
except Exception:
    ProfilerActivity = None
    torch_profile = None
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Set up logger
logger = logging.getLogger(__name__)


class _StderrSuppressor:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._saved_fd = None

    def __enter__(self):
        if not self.enabled:
            return self
        self._saved_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 2)
            os.close(self._saved_fd)
            self._saved_fd = None
        return False


def _profile_flops(fn: Callable[[], Any], silent: bool = False) -> Tuple[Any, float]:
    if torch_profile is None or ProfilerActivity is None:
        return fn(), 0.0
    with _StderrSuppressor(silent), torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        result = fn()
    total_flops = 0.0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", None)
        if flops:
            total_flops += float(flops)
    return result, total_flops


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


@dataclass
class TokenSelectionConfig:
    """Configuration for gradient-based token importance, selection, and pruning."""

    token_selection_enabled: bool = False
    token_prune_enabled: bool = False
    vision_partial_update_enabled: bool = False
    flops_profile_enabled: bool = False
    flops_profile_silent: bool = True
    region_eval_interval: int = 1
    grad_denoise_steps: int = 1
    grad_region_mass: float = 0.25
    grad_region_ema: Optional[float] = None
    grad_keep_prev: bool = False
    grad_score_method: str = "full_grad"
    partial_grad_phi: str = "l2"
    partial_grad_pos_weight: float = 1.0
    partial_grad_grip_weight: float = 2.0
    attn_score_beta: float = 1.0
    head_consensus_gating_enabled: bool = False
    head_consensus_eta: float = 1.0
    head_g_beta: float = 1.0
    head_g_norm: str = "sum"
    token_reuse_mode: str = "none"
    grad_tau: float = 0.1
    grad_alpha: float = 1.0
    grad_beta: float = 1.0
    token_temporal_threshold: float = 0.9
    token_spatial_threshold: float = 0.9
    token_spatial_radius: int = 1
    min_kept_tokens: int = 1
    max_kept_tokens: Optional[int] = None
    region_patch_size: int = 1


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    @staticmethod
    def _estimate_transformer_flops(n_tokens: int, hidden_dim: int, mlp_dim: int, num_layers: int, batch: int) -> float:
        if n_tokens <= 1 or hidden_dim <= 0 or mlp_dim <= 0 or num_layers <= 0 or batch <= 0:
            return 0.0
        n = float(n_tokens)
        d = float(hidden_dim)
        m = float(mlp_dim)
        return float(batch) * float(num_layers) * (4.0 * n * d * d + 2.0 * n * n * d + 3.0 * n * d * m)

    def _estimate_llm_flops(self, seq_len: int, batch: int) -> float:
        cfg = getattr(self.language_model, "config", None)
        if cfg is None:
            return 0.0
        hidden_dim = int(getattr(cfg, "hidden_size", 0) or 0)
        mlp_dim = int(getattr(cfg, "intermediate_size", 0) or 0)
        if mlp_dim <= 0 and hidden_dim > 0:
            mlp_dim = 4 * hidden_dim
        num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        return self._estimate_transformer_flops(seq_len, hidden_dim, mlp_dim, num_layers, batch)

    def _estimate_vit_flops(self, model: nn.Module, tokens_per_image: int, num_images: int, batch: int) -> float:
        if model is None:
            return 0.0
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            return 0.0
        hidden_dim = int(getattr(model, "embed_dim", 0) or 0)
        if hidden_dim <= 0:
            return 0.0
        block0 = blocks[0]
        mlp = getattr(block0, "mlp", None)
        mlp_dim = 0
        if mlp is not None:
            fc1 = getattr(mlp, "fc1", None)
            if fc1 is not None and hasattr(fc1, "out_features"):
                mlp_dim = int(fc1.out_features)
            elif hasattr(mlp, "hidden_features"):
                mlp_dim = int(mlp.hidden_features)
        if mlp_dim <= 0:
            return 0.0
        num_layers = len(blocks)
        return self._estimate_transformer_flops(tokens_per_image, hidden_dim, mlp_dim, num_layers, batch * num_images)

    def _estimate_projector_flops(self, tokens: int, batch: int) -> float:
        if tokens <= 0 or batch <= 0:
            return 0.0
        flops = 0.0
        for layer_name in ("fc1", "fc2", "fc3"):
            layer = getattr(self.projector, layer_name, None)
            if layer is None:
                continue
            if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                flops += 2.0 * float(batch) * float(tokens) * float(layer.in_features) * float(layer.out_features)
        return flops

    def _estimate_vision_total_flops(self, batch: int) -> float:
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        if num_images <= 0 or num_patches <= 0 or batch <= 0:
            return 0.0
        featurizer = getattr(self.vision_backbone, "featurizer", None)
        prefix = int(getattr(featurizer, "num_prefix_tokens", 0) or 0) if featurizer is not None else 0
        tokens_per_image = num_patches + max(prefix, 0)
        flops = self._estimate_vit_flops(featurizer, tokens_per_image, num_images, batch)
        if getattr(self.vision_backbone, "use_fused_vision_backbone", False):
            fused = getattr(self.vision_backbone, "fused_featurizer", None)
            prefix_fused = int(getattr(fused, "num_prefix_tokens", 0) or 0) if fused is not None else 0
            tokens_per_image_fused = num_patches + max(prefix_fused, 0)
            flops += self._estimate_vit_flops(fused, tokens_per_image_fused, num_images, batch)
        flops += self._estimate_projector_flops(num_patches * num_images, batch)
        return flops

    def _estimate_action_head_flops(self, action_head: nn.Module, num_chunks: int, batch: int) -> float:
        if action_head is None or num_chunks <= 0 or batch <= 0:
            return 0.0
        mlp = None
        if hasattr(action_head, "model"):
            mlp = getattr(action_head, "model", None)
        if mlp is None and hasattr(action_head, "noise_predictor"):
            noise_pred = getattr(action_head, "noise_predictor", None)
            if noise_pred is not None and hasattr(noise_pred, "mlp_resnet"):
                mlp = noise_pred.mlp_resnet
        if mlp is None:
            return 0.0
        fc1 = getattr(mlp, "fc1", None)
        fc2 = getattr(mlp, "fc2", None)
        if fc1 is None or fc2 is None:
            return 0.0
        in_dim = int(getattr(fc1, "in_features", 0) or 0)
        hidden_dim = int(getattr(fc1, "out_features", 0) or 0)
        out_dim = int(getattr(fc2, "out_features", 0) or 0)
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            return 0.0
        flops_per = 2.0 * in_dim * hidden_dim + 2.0 * hidden_dim * out_dim
        blocks = getattr(mlp, "mlp_resnet_blocks", None)
        if blocks is not None:
            for block in blocks:
                linear = getattr(block, "ffn", None)
                if linear is not None and len(linear) > 1:
                    layer = linear[1]
                    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
                        flops_per += 2.0 * float(layer.in_features) * float(layer.out_features)
        return float(batch) * float(num_chunks) * flops_per

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """Build multimodal embeddings and attention mask"""
        # Update attention mask
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # Get visual features
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # Add proprioceptive state if provided
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

            # [Diffusion] Add diffusion timestep embedding if provided
            if diffusion_timestep_embeddings is not None:
                # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
                projected_patch_embeddings = torch.cat(
                    (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
                )

            # Process action embeddings
            if noisy_actions is not None:
                # Get mask corresponding to all action tokens
                all_actions_mask = self._process_action_masks(labels)

                # Reshape noisy actions into individual action tokens
                # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
                B = noisy_actions.shape[0]
                noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

                # Project noisy action tokens into language model embedding space
                noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

                # Replace embeddings of the action tokens with noisy action embeddings
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, noisy_action_features
                )
            else:
                # Replace the embeddings of the action tokens with zeros
                # (Later on, the positional embeddings will be added to them)
                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
                input_embeddings = input_embeddings * ~all_actions_mask

            # Build multimodal embeddings & attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Build labels for multimodal sequence if needed
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

        self._token_selection_cfg: Optional[TokenSelectionConfig] = None
        self.reset_token_selection_state()

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def configure_token_selection(self, cfg: Optional[TokenSelectionConfig]) -> None:
        """Configure token selection and reset cached state."""
        self._token_selection_cfg = cfg
        self.reset_token_selection_state()

    def reset_token_selection_state(self) -> None:
        """Reset per-episode token selection state."""
        self._token_selection_state: Dict[str, Any] = {
            "frame_idx": 0,
            "last_image_embs": None,
            "last_important_mask": None,
            "last_important_region_mask": None,
            "last_keep_mask": None,
            "last_keep_pre_mask": None,
            "last_clipped_mask": None,
            "last_background_mask": None,
            "last_token_scores": None,
            "last_region_scores": None,
            "last_overlay_labels": None,
            "last_overlay_grid": None,
            "last_effective_important_mask": None,
            "last_eval_frame": False,
            "last_timing": None,
            "last_flops": None,
            "last_gate": None,
            "last_lambda_pos": None,
            "last_lambda_grip": None,
        }
        self._clear_llm_reuse_context()
        self._reset_llm_reuse_cache()

    def get_token_selection_state(self) -> Dict[str, Any]:
        """Expose last token selection state for debugging/visualization."""
        return self._token_selection_state

    def _get_llama_layers(self):
        lm = getattr(self, "language_model", None)
        if lm is None:
            return None
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
        return None

    def _ensure_llama_token_reuse_patch(self) -> None:
        if getattr(self, "_llama_reuse_patched", False):
            return
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        except Exception as exc:  # pragma: no cover - depends on runtime install
            logger.warning("Token reuse patch skipped (missing Llama): %s", exc)
            self._llama_reuse_patched = False
            return

        layers = self._get_llama_layers()
        if not layers:
            logger.warning("Token reuse patch skipped (no Llama layers found).")
            self._llama_reuse_patched = False
            return

        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def _apply_rotary_subset(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (q * cos) + (_rotate_half(q) * sin)

        def _reuse_attention_forward(
            self_attn,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Any] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            reuse_ctx = getattr(self_attn, "_token_reuse_context", None)
            if reuse_ctx is None or not reuse_ctx.get("enabled", False):
                return self_attn._orig_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            if getattr(self_attn.config, "pretraining_tp", 1) > 1:
                return self_attn._orig_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            reuse_mode = reuse_ctx.get("mode", "none")
            reuse_mask = reuse_ctx.get("reuse_mask", None)
            bsz, q_len, _ = hidden_states.size()
            if reuse_mask is not None:
                if reuse_mask.shape[:2] != (bsz, q_len):
                    reuse_mask = None
            cache = getattr(self_attn, "_token_reuse_cache", None)
            if reuse_mask is not None:
                if cache is None or cache.get("k") is None or cache.get("v") is None:
                    reuse_mask = None

            update_mask = None
            update_idx = None
            if reuse_mask is not None:
                update_mask = ~reuse_mask
                if bsz != 1:
                    reuse_mask = None
                    update_mask = None
                else:
                    update_idx = torch.nonzero(update_mask[0], as_tuple=False).squeeze(-1)

            if reuse_mode == "reuse_all" and update_idx is not None:
                if update_idx.numel() == 0:
                    attn_output = hidden_states.new_zeros(hidden_states.shape)
                    return attn_output, None, past_key_value
                hidden_update = hidden_states[:, update_idx, :]
                query_states = self_attn.q_proj(hidden_update)
                key_update = self_attn.k_proj(hidden_update)
                value_update = self_attn.v_proj(hidden_update)
                key_states = torch.zeros(
                    bsz,
                    q_len,
                    self_attn.num_key_value_heads * self_attn.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                value_states = torch.zeros_like(key_states)
                key_states[:, update_idx, :] = key_update
                value_states[:, update_idx, :] = value_update
            elif reuse_mode == "reuse_kv" and update_mask is not None:
                query_states = self_attn.q_proj(hidden_states)
                key_states = torch.zeros(
                    bsz,
                    q_len,
                    self_attn.num_key_value_heads * self_attn.head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                value_states = torch.zeros_like(key_states)
                if update_mask.any():
                    hidden_update = hidden_states[update_mask]
                    key_update = self_attn.k_proj(hidden_update)
                    value_update = self_attn.v_proj(hidden_update)
                    key_states[update_mask] = key_update
                    value_states[update_mask] = value_update
            else:
                query_states = self_attn.q_proj(hidden_states)
                key_states = self_attn.k_proj(hidden_states)
                value_states = self_attn.v_proj(hidden_states)

            if reuse_mode == "reuse_all" and update_idx is not None:
                query_states = query_states.view(bsz, update_idx.numel(), self_attn.num_heads, self_attn.head_dim)
                query_states = query_states.transpose(1, 2)
            else:
                query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(
                1, 2
            )

            cos, sin = self_attn.rotary_emb(value_states, position_ids)
            if reuse_mode == "reuse_all" and update_idx is not None:
                cos_u = cos[:, update_idx, :]
                sin_u = sin[:, update_idx, :]
                query_states = _apply_rotary_subset(query_states, cos_u, sin_u)
                key_states = apply_rotary_pos_emb(
                    key_states, key_states, cos, sin
                )[1]
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if reuse_mask is not None and cache is not None:
                cached_k = cache.get("k", None)
                cached_v = cache.get("v", None)
                if cached_k is not None and cached_k.shape == key_states.shape:
                    reuse_mask_full = reuse_mask.unsqueeze(1).unsqueeze(-1)
                    key_states = torch.where(reuse_mask_full, cached_k, key_states)
                    value_states = torch.where(reuse_mask_full, cached_v, value_states)

            if cache is None:
                cache = {}
                self_attn._token_reuse_cache = cache
            cache["k"] = key_states.detach()
            cache["v"] = value_states.detach()

            key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
            value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

            if attention_mask is None or (torch.is_tensor(attention_mask) and attention_mask.dim() == 2):
                min_dtype = torch.finfo(query_states.dtype).min
                causal = torch.full(
                    (q_len, key_states.shape[-2]),
                    fill_value=min_dtype,
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                causal = torch.triu(causal, diagonal=1)
                causal = causal.unsqueeze(0).unsqueeze(0)
                if torch.is_tensor(attention_mask) and attention_mask is not None and attention_mask.dim() == 2:
                    pad = (attention_mask == 0).to(query_states.dtype) * min_dtype
                    pad = pad.unsqueeze(1).unsqueeze(2)
                    causal = causal + pad
                attention_mask = causal

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
            if attention_mask is not None:
                if reuse_mode == "reuse_all" and update_idx is not None:
                    causal_mask = attention_mask[:, :, update_idx, : key_states.shape[-2]]
                else:
                    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if reuse_mode == "reuse_all" and update_idx is not None:
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, update_idx.numel(), self_attn.hidden_size)
            else:
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)

            attn_output = self_attn.o_proj(attn_output)

            if reuse_mode == "reuse_all" and update_idx is not None:
                full_output = torch.zeros(
                    bsz, q_len, self_attn.hidden_size, device=attn_output.device, dtype=attn_output.dtype
                )
                full_output[:, update_idx, :] = attn_output
                attn_output = full_output

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        def _reuse_layer_forward(
            self_layer,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            reuse_ctx = getattr(self_layer, "_token_reuse_context", None)
            if reuse_ctx is None or reuse_ctx.get("mode", "none") != "reuse_all":
                return self_layer._orig_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            reuse_mask = reuse_ctx.get("reuse_mask", None)
            if reuse_mask is None or reuse_mask.shape[:2] != hidden_states.shape[:2] or hidden_states.shape[0] != 1:
                return self_layer._orig_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            update_idx = torch.nonzero(~reuse_mask[0], as_tuple=False).squeeze(-1)
            cache = getattr(self_layer, "_token_reuse_cache", None)
            cached_input = None if cache is None else cache.get("input", None)
            cached_output = None if cache is None else cache.get("output", None)
            if cached_input is not None and cached_input.shape == hidden_states.shape:
                hidden_states = torch.where(reuse_mask.unsqueeze(-1), cached_input, hidden_states)

            input_cache = hidden_states.detach()
            residual = hidden_states
            hidden_states = self_layer.input_layernorm(hidden_states)

            hidden_states, self_attn_weights, present_key_value = self_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self_layer.post_attention_layernorm(hidden_states)
            if update_idx.numel() > 0:
                update_states = hidden_states[:, update_idx, :]
                update_states = self_layer.mlp(update_states)
                hidden_states = hidden_states.clone()
                hidden_states[:, update_idx, :] = residual[:, update_idx, :] + update_states
            else:
                hidden_states = residual

            if cached_output is not None and cached_output.shape == hidden_states.shape:
                hidden_states = torch.where(reuse_mask.unsqueeze(-1), cached_output, hidden_states)

            if cache is None:
                cache = {}
                self_layer._token_reuse_cache = cache
            cache["input"] = input_cache
            cache["output"] = hidden_states.detach()

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        for layer in layers:
            if hasattr(layer, "self_attn") and not hasattr(layer.self_attn, "_orig_forward"):
                layer.self_attn._orig_forward = layer.self_attn.forward
                layer.self_attn.forward = types.MethodType(_reuse_attention_forward, layer.self_attn)
            if not hasattr(layer, "_orig_forward"):
                layer._orig_forward = layer.forward
                layer.forward = types.MethodType(_reuse_layer_forward, layer)

        self._llama_reuse_patched = True

    def _set_llm_reuse_context(
        self,
        reuse_mask: Optional[torch.Tensor],
        reuse_mode: str,
        force: bool = False,
    ) -> None:
        mode = str(reuse_mode).lower()
        if mode == "none" and not force:
            return
        self._ensure_llama_token_reuse_patch()
        layers = self._get_llama_layers()
        if not layers:
            return
        ctx = {"enabled": True, "mode": mode, "reuse_mask": reuse_mask}
        for layer in layers:
            layer._token_reuse_context = ctx
            if hasattr(layer, "self_attn"):
                layer.self_attn._token_reuse_context = ctx

    def _clear_llm_reuse_context(self) -> None:
        layers = self._get_llama_layers()
        if not layers:
            return
        for layer in layers:
            if hasattr(layer, "_token_reuse_context"):
                layer._token_reuse_context = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "_token_reuse_context"):
                layer.self_attn._token_reuse_context = None

    def _reset_llm_reuse_cache(self) -> None:
        layers = self._get_llama_layers()
        if not layers:
            return
        for layer in layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn._token_reuse_cache = {"k": None, "v": None}
            layer._token_reuse_cache = {"input": None, "output": None}

    def _build_llm_reuse_mask(
        self,
        keep_mask: torch.Tensor,
        num_patch_tokens: int,
        input_embeddings: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if keep_mask is None:
            return None
        if keep_mask.dim() == 1:
            keep_mask = keep_mask.unsqueeze(0)
        reuse_patch = ~keep_mask
        if reuse_patch.shape[1] > num_patch_tokens:
            reuse_patch = reuse_patch[:, :num_patch_tokens]
        elif reuse_patch.shape[1] < num_patch_tokens:
            pad = num_patch_tokens - reuse_patch.shape[1]
            pad_mask = torch.zeros(
                (reuse_patch.shape[0], pad), device=reuse_patch.device, dtype=torch.bool
            )
            reuse_patch = torch.cat([reuse_patch, pad_mask], dim=1)
        full_len = 1 + num_patch_tokens + (input_embeddings.shape[1] - 1)
        reuse_full = torch.zeros(
            (reuse_patch.shape[0], full_len), device=reuse_patch.device, dtype=torch.bool
        )
        reuse_full[:, 1 : 1 + num_patch_tokens] = reuse_patch
        return reuse_full

    def _resolve_token_selection_cfg(self, cfg: Optional[TokenSelectionConfig]) -> Optional[TokenSelectionConfig]:
        if cfg is not None:
            return cfg
        return self._token_selection_cfg

    def _get_patch_grid_size(self, num_patches: Optional[int] = None) -> Tuple[int, int]:
        num_patches = num_patches or self.vision_backbone.get_num_patches()
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Expected square patch grid, got num_patches={num_patches}")
        return grid_size, grid_size

    def _reshape_tokens_to_grid(
        self,
        token_values: torch.Tensor,
        num_images: int,
        num_patches: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if token_values.dim() == 1:
            token_values = token_values.unsqueeze(0)
        B = token_values.shape[0]
        grids = []
        for img_idx in range(num_images):
            start = img_idx * num_patches
            end = start + num_patches
            grids.append(token_values[:, start:end].reshape(B, grid_h, grid_w))
        return torch.stack(grids, dim=1)

    def _compute_region_scores(
        self,
        token_scores: torch.Tensor,
        region_patch_size: int,
    ) -> torch.Tensor:
        if token_scores.dim() == 1:
            token_scores = token_scores.unsqueeze(0)
        B, total_tokens = token_scores.shape
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        if total_tokens != num_images * num_patches:
            raise ValueError("Token score shape does not match current vision token count.")
        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        region_patch_size = max(1, int(region_patch_size))
        if (grid_h % region_patch_size) != 0 or (grid_w % region_patch_size) != 0:
            raise ValueError("region_patch_size must evenly divide patch grid dimensions.")
        region_h = grid_h // region_patch_size
        region_w = grid_w // region_patch_size

        region_scores = torch.zeros(
            (B, num_images, region_h, region_w), device=token_scores.device, dtype=token_scores.dtype
        )
        for img_idx in range(num_images):
            start = img_idx * num_patches
            end = start + num_patches
            scores_img = token_scores[:, start:end].reshape(B, grid_h, grid_w)
            if region_patch_size == 1:
                region_scores[:, img_idx] = scores_img
            else:
                scores_regions = scores_img.reshape(B, region_h, region_patch_size, region_w, region_patch_size)
                region_scores[:, img_idx] = scores_regions.mean(dim=(2, 4))

        return region_scores

    def _select_important_mask(
        self,
        region_scores: torch.Tensor,
        grad_region_mass: float,
        region_patch_size: int,
        return_region_mask: bool = False,
    ) -> torch.Tensor:
        B, num_images, region_h, region_w = region_scores.shape
        num_patches = self.vision_backbone.get_num_patches()
        total_tokens = num_patches * num_images
        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        region_patch_size = max(1, int(region_patch_size))
        region_mask = torch.zeros(
            (B, num_images * region_h * region_w), device=region_scores.device, dtype=torch.bool
        )
        if return_region_mask:
            for b in range(B):
                for img_idx in range(num_images):
                    scores_flat = region_scores[b, img_idx].reshape(-1)
                    if scores_flat.numel() == 0:
                        continue
                    total = scores_flat.sum()
                    if total <= 0:
                        normalized = torch.full_like(scores_flat, 1.0 / scores_flat.numel())
                    else:
                        normalized = scores_flat / total
                    sorted_scores, sorted_idx = torch.sort(normalized, descending=True)
                    keep_count = int((sorted_scores.cumsum(dim=0) < grad_region_mass).sum().item()) + 1
                    keep_count = min(max(keep_count, 1), sorted_scores.numel())
                    region_keep = torch.zeros_like(scores_flat, dtype=torch.bool)
                    region_keep[sorted_idx[:keep_count]] = True
                    start = img_idx * region_h * region_w
                    region_mask[b, start : start + region_h * region_w] = region_keep
            return region_mask

        token_mask = torch.zeros((B, total_tokens), device=region_scores.device, dtype=torch.bool)
        for b in range(B):
            for img_idx in range(num_images):
                scores_flat = region_scores[b, img_idx].reshape(-1)
                if scores_flat.numel() == 0:
                    continue
                total = scores_flat.sum()
                if total <= 0:
                    normalized = torch.full_like(scores_flat, 1.0 / scores_flat.numel())
                else:
                    normalized = scores_flat / total
                sorted_scores, sorted_idx = torch.sort(normalized, descending=True)
                keep_count = int((sorted_scores.cumsum(dim=0) < grad_region_mass).sum().item()) + 1
                keep_count = min(max(keep_count, 1), sorted_scores.numel())
                region_keep = torch.zeros_like(scores_flat, dtype=torch.bool)
                region_keep[sorted_idx[:keep_count]] = True
                region_keep_grid = region_keep.reshape(region_h, region_w)
                token_mask_grid = region_keep_grid.repeat_interleave(region_patch_size, dim=0).repeat_interleave(
                    region_patch_size, dim=1
                )
                token_mask_flat = token_mask_grid.reshape(-1)
                start = img_idx * num_patches
                token_mask[b, start : start + num_patches] = token_mask_flat

        return token_mask

    def _expand_region_mask(self, region_mask: torch.Tensor, region_patch_size: int) -> torch.Tensor:
        if region_mask.dim() == 1:
            region_mask = region_mask.unsqueeze(0)
        B, total_regions = region_mask.shape
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        region_patch_size = max(1, int(region_patch_size))
        region_h = grid_h // region_patch_size
        region_w = grid_w // region_patch_size
        expected_regions = num_images * region_h * region_w
        if total_regions != expected_regions:
            raise ValueError("Region mask shape does not match expected region count.")
        token_mask = torch.zeros((B, num_images * num_patches), device=region_mask.device, dtype=torch.bool)
        for img_idx in range(num_images):
            start_r = img_idx * region_h * region_w
            end_r = start_r + region_h * region_w
            region_grid = region_mask[:, start_r:end_r].reshape(B, region_h, region_w)
            token_grid = region_grid.repeat_interleave(region_patch_size, dim=1).repeat_interleave(
                region_patch_size, dim=2
            )
            start_t = img_idx * num_patches
            end_t = start_t + num_patches
            token_mask[:, start_t:end_t] = token_grid.reshape(B, -1)
        return token_mask

    def _compute_background_region_mask(
        self,
        current_tokens: torch.Tensor,
        last_tokens: Optional[torch.Tensor],
        token_temporal_threshold: float,
        token_spatial_threshold: float,
        token_spatial_radius: int,
        region_patch_size: int,
    ) -> torch.Tensor:
        if current_tokens.dim() == 2:
            current_tokens = current_tokens.unsqueeze(0)
        current_tokens_f = current_tokens.float()
        last_tokens_f = None if last_tokens is None else last_tokens.float()
        B, total_tokens, _ = current_tokens.shape
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        if total_tokens != num_images * num_patches:
            raise ValueError("Token shape does not match current vision token count.")
        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        region_patch_size = max(1, int(region_patch_size))
        if (grid_h % region_patch_size) != 0 or (grid_w % region_patch_size) != 0:
            raise ValueError("region_patch_size must evenly divide patch grid dimensions.")
        region_h = grid_h // region_patch_size
        region_w = grid_w // region_patch_size

        def _pool_regions(tokens: torch.Tensor) -> torch.Tensor:
            regions = []
            for img_idx in range(num_images):
                start = img_idx * num_patches
                end = start + num_patches
                tokens_img = tokens[:, start:end].reshape(B, grid_h, grid_w, -1)
                pooled = tokens_img.reshape(
                    B, region_h, region_patch_size, region_w, region_patch_size, -1
                ).mean(dim=(2, 4))
                regions.append(pooled)
            return torch.stack(regions, dim=1)

        current_regions = _pool_regions(current_tokens_f)
        last_regions = None if last_tokens_f is None else _pool_regions(last_tokens_f)

        temporal_mask = torch.zeros(
            (B, num_images, region_h, region_w), device=current_tokens.device, dtype=torch.bool
        )
        if last_regions is not None and last_regions.shape == current_regions.shape:
            temporal_sim = F.cosine_similarity(current_regions, last_regions, dim=-1)
            temporal_mask = temporal_sim >= token_temporal_threshold

        token_spatial_radius = max(0, int(token_spatial_radius))
        spatial_mask = torch.zeros_like(temporal_mask, dtype=torch.bool)
        norm_regions = F.normalize(current_regions, dim=-1)
        for img_idx in range(num_images):
            regions_img = norm_regions[:, img_idx]
            for h in range(region_h):
                h0 = max(0, h - token_spatial_radius)
                h1 = min(region_h, h + token_spatial_radius + 1)
                for w in range(region_w):
                    w0 = max(0, w - token_spatial_radius)
                    w1 = min(region_w, w + token_spatial_radius + 1)
                    neighbors = regions_img[:, h0:h1, w0:w1, :].reshape(B, -1, regions_img.shape[-1])
                    if neighbors.shape[1] == 1:
                        spatial_sim = torch.ones(B, device=current_tokens.device, dtype=current_tokens.dtype)
                    else:
                        sim = (neighbors * regions_img[:, h, w, :].unsqueeze(1)).sum(dim=-1)
                        self_idx = (h - h0) * (w1 - w0) + (w - w0)
                        sim = torch.cat([sim[:, :self_idx], sim[:, self_idx + 1 :]], dim=1)
                        spatial_sim = sim.mean(dim=1)
                    spatial_mask[:, img_idx, h, w] = spatial_sim >= token_spatial_threshold

        background = temporal_mask & spatial_mask
        background_flat = torch.zeros(
            (B, num_images * region_h * region_w), device=current_tokens.device, dtype=torch.bool
        )
        for img_idx in range(num_images):
            start = img_idx * region_h * region_w
            end = start + region_h * region_w
            background_flat[:, start:end] = background[:, img_idx].reshape(B, -1)
        return background_flat

    def _compute_background_mask(
        self,
        current_tokens: torch.Tensor,
        last_tokens: Optional[torch.Tensor],
        token_temporal_threshold: float,
        token_spatial_threshold: float,
        token_spatial_radius: int,
    ) -> torch.Tensor:
        if current_tokens.dim() == 2:
            current_tokens = current_tokens.unsqueeze(0)
        # Avoid unsupported bf16 ops in similarity math.
        current_tokens_f = current_tokens.float()
        last_tokens_f = None if last_tokens is None else last_tokens.float()
        B, total_tokens, _ = current_tokens.shape
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        if total_tokens != num_images * num_patches:
            raise ValueError("Token shape does not match current vision token count.")

        temporal_mask = torch.zeros((B, total_tokens), device=current_tokens.device, dtype=torch.bool)
        if last_tokens_f is not None and last_tokens_f.shape == current_tokens_f.shape:
            temporal_sim = F.cosine_similarity(current_tokens_f, last_tokens_f, dim=-1)
            temporal_mask = temporal_sim >= token_temporal_threshold

        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        token_spatial_radius = max(0, int(token_spatial_radius))
        spatial_mask = torch.zeros((B, total_tokens), device=current_tokens.device, dtype=torch.bool)
        norm_tokens = F.normalize(current_tokens_f, dim=-1)

        for img_idx in range(num_images):
            start = img_idx * num_patches
            end = start + num_patches
            tokens_img = norm_tokens[:, start:end].reshape(B, grid_h, grid_w, -1)
            for h in range(grid_h):
                h0 = max(0, h - token_spatial_radius)
                h1 = min(grid_h, h + token_spatial_radius + 1)
                for w in range(grid_w):
                    w0 = max(0, w - token_spatial_radius)
                    w1 = min(grid_w, w + token_spatial_radius + 1)
                    neighbors = tokens_img[:, h0:h1, w0:w1, :].reshape(B, -1, tokens_img.shape[-1])
                    if neighbors.shape[1] == 1:
                        spatial_sim = torch.ones(B, device=current_tokens.device, dtype=current_tokens.dtype)
                    else:
                        sim = (neighbors * tokens_img[:, h, w, :].unsqueeze(1)).sum(dim=-1)
                        self_idx = (h - h0) * (w1 - w0) + (w - w0)
                        sim = torch.cat([sim[:, :self_idx], sim[:, self_idx + 1 :]], dim=1)
                        spatial_sim = sim.mean(dim=1)
                    idx = start + h * grid_w + w
                    spatial_mask[:, idx] = spatial_sim >= token_spatial_threshold

        token_valid = torch.ones_like(spatial_mask, dtype=torch.bool)
        return temporal_mask & spatial_mask & token_valid

    def _apply_keep_constraints(
        self,
        keep_pre: torch.Tensor,
        token_scores: torch.Tensor,
        min_kept_tokens: int,
        max_kept_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keep_pre.dim() == 1:
            keep_pre = keep_pre.unsqueeze(0)
        if token_scores.dim() == 1:
            token_scores = token_scores.unsqueeze(0)
        B, total_tokens = keep_pre.shape
        keep_final = keep_pre.clone()
        clipped_mask = torch.zeros_like(keep_pre, dtype=torch.bool)

        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        expected_tokens = num_images * num_patches
        per_image_min = None
        per_image_max = None
        if total_tokens == expected_tokens:
            per_image_min = max(0, int(min_kept_tokens))
            per_image_min = min(per_image_min, num_patches)
            if max_kept_tokens is not None:
                per_image_max = int(max_kept_tokens)
                if per_image_max <= 0:
                    per_image_max = num_patches
                else:
                    per_image_max = min(per_image_max, num_patches)
                if per_image_max < per_image_min:
                    per_image_max = per_image_min
        use_per_image = total_tokens == expected_tokens and (
            (per_image_min is not None and per_image_min > 0) or per_image_max is not None
        )

        for b in range(B):
            keep = keep_final[b].clone()
            scores = token_scores[b]
            min_kept = max(0, int(min_kept_tokens))
            min_kept = min(min_kept, total_tokens)

            if not use_per_image:
                max_kept = total_tokens if max_kept_tokens is None else int(max_kept_tokens)
                max_kept = max(1, max_kept) if max_kept > 0 else total_tokens
                max_kept = min(max_kept, total_tokens)
                if min_kept > max_kept:
                    max_kept = min_kept

                if keep.sum().item() < min_kept:
                    needed = min_kept - int(keep.sum().item())
                    scores_fill = scores.clone()
                    scores_fill[keep] = float("-inf")
                    _, add_idx = torch.topk(scores_fill, k=needed)
                    keep[add_idx] = True

                if keep.sum().item() > max_kept:
                    scores_fill = scores.clone()
                    scores_fill[~keep] = float("-inf")
                    _, keep_idx = torch.topk(scores_fill, k=max_kept)
                    new_keep = torch.zeros_like(keep, dtype=torch.bool)
                    new_keep[keep_idx] = True
                    keep = new_keep

                keep_final[b] = keep
                clipped_mask[b] = keep_pre[b] & ~keep
                continue

            for img_idx in range(num_images):
                start = img_idx * num_patches
                end = start + num_patches
                keep_img = keep[start:end]
                if per_image_min is not None and keep_img.sum().item() < per_image_min:
                    needed = per_image_min - int(keep_img.sum().item())
                    scores_img = scores[start:end]
                    scores_fill = scores_img.clone()
                    scores_fill[keep_img] = float("-inf")
                    _, add_idx = torch.topk(scores_fill, k=needed)
                    keep_img[add_idx] = True
                if per_image_max is not None and keep_img.sum().item() > per_image_max:
                    scores_img = scores[start:end]
                    scores_fill = scores_img.clone()
                    scores_fill[~keep_img] = float("-inf")
                    _, keep_idx = torch.topk(scores_fill, k=per_image_max)
                    new_keep_img = torch.zeros_like(keep_img, dtype=torch.bool)
                    new_keep_img[keep_idx] = True
                    keep[start:end] = new_keep_img

            keep_final[b] = keep
            clipped_mask[b] = keep_pre[b] & ~keep

        return keep_final, clipped_mask

    def _apply_keep_constraints_regions(
        self,
        keep_pre: torch.Tensor,
        region_scores: torch.Tensor,
        min_kept_tokens: int,
        max_kept_tokens: Optional[int],
        region_patch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keep_pre.dim() == 1:
            keep_pre = keep_pre.unsqueeze(0)
        if region_scores.dim() == 4:
            region_scores = region_scores.reshape(region_scores.shape[0], -1)
        if region_scores.dim() == 1:
            region_scores = region_scores.unsqueeze(0)
        B, total_regions = keep_pre.shape
        keep_final = keep_pre.clone()
        clipped_mask = torch.zeros_like(keep_pre, dtype=torch.bool)

        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        grid_h, grid_w = self._get_patch_grid_size(num_patches)
        region_patch_size = max(1, int(region_patch_size))
        region_h = grid_h // region_patch_size
        region_w = grid_w // region_patch_size
        regions_per_image = region_h * region_w
        expected_regions = num_images * regions_per_image
        if total_regions != expected_regions:
            raise ValueError("Region mask shape does not match expected region count.")

        region_area = region_patch_size * region_patch_size
        per_image_min = 0
        if min_kept_tokens > 0:
            per_image_min = int(math.ceil(float(min_kept_tokens) / float(region_area)))
            per_image_min = max(1, per_image_min)
            per_image_min = min(per_image_min, regions_per_image)
        per_image_max = None
        if max_kept_tokens is not None:
            per_image_max = int(max_kept_tokens) // region_area
            if per_image_max <= 0:
                per_image_max = 1
            per_image_max = min(per_image_max, regions_per_image)
            if per_image_max < per_image_min:
                per_image_max = per_image_min

        use_per_image = expected_regions == total_regions and (per_image_min > 0 or per_image_max is not None)

        for b in range(B):
            keep = keep_final[b].clone()
            scores = region_scores[b]
            min_kept = per_image_min if use_per_image else max(0, int(min_kept_tokens))
            min_kept = min(min_kept, total_regions)
            max_kept = total_regions
            if not use_per_image:
                if max_kept_tokens is not None:
                    max_kept = int(max_kept_tokens)
                    if max_kept <= 0:
                        max_kept = total_regions
                    max_kept = min(max_kept, total_regions)
                if min_kept > max_kept:
                    max_kept = min_kept

            if not use_per_image:
                if min_kept > 0 and keep.sum().item() < min_kept:
                    needed = min_kept - int(keep.sum().item())
                    scores_fill = scores.clone()
                    scores_fill[keep] = float("-inf")
                    _, add_idx = torch.topk(scores_fill, k=needed)
                    keep[add_idx] = True
                if max_kept is not None and keep.sum().item() > max_kept:
                    scores_fill = scores.clone()
                    scores_fill[~keep] = float("-inf")
                    _, keep_idx = torch.topk(scores_fill, k=max_kept)
                    new_keep = torch.zeros_like(keep, dtype=torch.bool)
                    new_keep[keep_idx] = True
                    keep = new_keep
            else:
                for img_idx in range(num_images):
                    start = img_idx * regions_per_image
                    end = start + regions_per_image
                    keep_img = keep[start:end].clone()
                    scores_img = scores[start:end]
                    if per_image_min > 0 and keep_img.sum().item() < per_image_min:
                        needed = per_image_min - int(keep_img.sum().item())
                        scores_fill = scores_img.clone()
                        scores_fill[keep_img] = float("-inf")
                        _, add_idx = torch.topk(scores_fill, k=needed)
                        keep_img[add_idx] = True
                    if per_image_max is not None and keep_img.sum().item() > per_image_max:
                        scores_fill = scores_img.clone()
                        scores_fill[~keep_img] = float("-inf")
                        _, keep_idx = torch.topk(scores_fill, k=per_image_max)
                        new_keep_img = torch.zeros_like(keep_img, dtype=torch.bool)
                        new_keep_img[keep_idx] = True
                        keep_img = new_keep_img
                    keep[start:end] = keep_img

            keep_final[b] = keep
            clipped_mask[b] = keep_pre[b] & ~keep

        return keep_final, clipped_mask

    def _get_attn_out_proj_weight(self) -> Optional[torch.Tensor]:
        lm = self.language_model
        layer_candidates = [
            ("model", "layers"),
            ("base_model", "model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("model", "h"),
        ]
        layers = None
        for path in layer_candidates:
            obj = lm
            for name in path:
                if not hasattr(obj, name):
                    obj = None
                    break
                obj = getattr(obj, name)
            if obj is not None:
                layers = obj
                break
        if not layers:
            return None
        last_layer = layers[-1]
        attn = getattr(last_layer, "self_attn", None)
        if attn is None:
            return None
        for proj_name in ("o_proj", "out_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                return proj.weight
        return None

    def _compute_partial_grad_scores(
        self,
        action_head,
        actions_hidden_states: torch.Tensor,
        action_pred: torch.Tensor,
        attn_weights: torch.Tensor,
        num_prompt_tokens: int,
        num_patches_with_extra: int,
        token_cfg: TokenSelectionConfig,
    ) -> Optional[torch.Tensor]:
        if (
            action_head is None
            or actions_hidden_states is None
            or action_pred is None
            or attn_weights is None
        ):
            return None
        if actions_hidden_states.dim() != 3 or action_pred.dim() != 3:
            return None
        w_o = self._get_attn_out_proj_weight()
        if w_o is None:
            logger.warning("Partial-grad scoring requires attention output projection; falling back.")
            return None

        B, action_tokens, hidden_dim = actions_hidden_states.shape
        num_heads = attn_weights.shape[1]
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            logger.warning("Partial-grad scoring has incompatible head dimensions; falling back.")
            return None

        phi = str(token_cfg.partial_grad_phi).lower()
        if phi == "l1":
            phi_prime = torch.sign(action_pred)
        else:
            phi_prime = 2.0 * action_pred

        lambda_weights = torch.full(
            (action_pred.shape[-1],),
            float(token_cfg.partial_grad_pos_weight),
            device=action_pred.device,
            dtype=action_pred.dtype,
        )
        lambda_weights[-1] = float(token_cfg.partial_grad_grip_weight)
        g_a = phi_prime * lambda_weights
        chunk_weights = torch.ones(
            (action_pred.shape[1],),
            device=action_pred.device,
            dtype=action_pred.dtype,
        )
        g_a = g_a * chunk_weights.view(1, -1, 1)

        with torch.enable_grad():
            z_p = actions_hidden_states.detach().requires_grad_(True)
            pred = action_head.predict_action(z_p)
            g_z = torch.autograd.grad(
                pred,
                z_p,
                grad_outputs=g_a,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
        if g_z is None:
            return None

        w_o = w_o.to(g_z.device)
        g_concat = torch.matmul(g_z.float(), w_o.float().t())
        head_dim = g_concat.shape[-1] // num_heads
        if head_dim * num_heads != g_concat.shape[-1]:
            return None
        g_head = g_concat.view(B, action_tokens, num_heads, head_dim)
        g_head_norm = torch.linalg.norm(g_head, dim=-1)
        norm_mode = str(getattr(token_cfg, "head_g_norm", "sum")).lower()
        if norm_mode not in {"sum", "max"}:
            norm_mode = "sum"
        if norm_mode == "max":
            denom = g_head_norm.max(dim=2, keepdim=True).values.clamp_min(1e-6)
        else:
            denom = g_head_norm.sum(dim=2, keepdim=True).clamp_min(1e-6)
        g_tilde = g_head_norm / denom
        if bool(getattr(token_cfg, "head_consensus_gating_enabled", False)):
            eta = float(getattr(token_cfg, "head_consensus_eta", 1.0))
            if eta != 1.0:
                g_tilde = g_tilde.clamp_min(0.0).pow(eta)
        g_beta = float(getattr(token_cfg, "head_g_beta", 1.0))
        if g_beta != 1.0:
            g_tilde = g_tilde.clamp_min(0.0).pow(g_beta)
        g_head_norm = g_tilde

        vision_count = (
            self.vision_backbone.get_num_images_in_input()
            * self.vision_backbone.get_num_patches()
        )
        action_start = int(num_patches_with_extra) + int(num_prompt_tokens)
        action_end = action_start + action_tokens
        vision_start = 1
        vision_end = vision_start + vision_count
        seq_len = attn_weights.shape[-1]
        if action_end > seq_len or vision_end > seq_len:
            return None

        attn_sel = attn_weights[:, :, action_start:action_end, vision_start:vision_end]
        attn_sel = attn_sel.float()
        beta = float(getattr(token_cfg, "attn_score_beta", 1.0))
        if beta != 1.0:
            attn_sel = attn_sel.clamp_min(0.0).pow(beta)
        g_norm = g_head_norm.permute(0, 2, 1).unsqueeze(-1)
        scores = (attn_sel * g_norm).sum(dim=(1, 2))
        return scores

    def _compute_attn_only_scores(
        self,
        attn_weights: torch.Tensor,
        action_tokens: int,
        num_prompt_tokens: int,
        num_patches_with_extra: int,
        token_cfg: TokenSelectionConfig,
    ) -> Optional[torch.Tensor]:
        if attn_weights is None or attn_weights.dim() != 4:
            return None
        num_images = self.vision_backbone.get_num_images_in_input()
        num_patches = self.vision_backbone.get_num_patches()
        vision_count = num_images * num_patches
        action_start = int(num_patches_with_extra) + int(num_prompt_tokens)
        action_end = action_start + int(action_tokens)
        vision_start = 1
        vision_end = vision_start + vision_count
        seq_len = attn_weights.shape[-1]
        if action_end > seq_len or vision_end > seq_len:
            return None
        attn_sel = attn_weights[:, :, action_start:action_end, vision_start:vision_end]
        attn_sel = attn_sel.float()
        beta = float(getattr(token_cfg, "attn_score_beta", 1.0))
        if beta != 1.0:
            attn_sel = attn_sel.clamp_min(0.0).pow(beta)
        return attn_sel.sum(dim=(1, 2))

    def _build_overlay_labels(
        self,
        keep_final: torch.Tensor,
        important_mask: torch.Tensor,
        clipped_mask: torch.Tensor,
    ) -> torch.Tensor:
        if keep_final.dim() == 1:
            keep_final = keep_final.unsqueeze(0)
        if important_mask.dim() == 1:
            important_mask = important_mask.unsqueeze(0)
        if clipped_mask.dim() == 1:
            clipped_mask = clipped_mask.unsqueeze(0)
        overlay = torch.zeros_like(keep_final, dtype=torch.uint8)
        overlay[keep_final & ~important_mask] = 1
        overlay[keep_final & important_mask] = 2
        overlay[clipped_mask] = 3
        return overlay

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
        collect_noise_preds: bool = False,
        grad_denoise_steps: int = 0,
        timing: Optional[Dict[str, float]] = None,
        timer_device: Optional[torch.device] = None,
        flops: Optional[Dict[str, float]] = None,
        profile_flops: bool = False,
        profile_flops_silent: bool = False,
    ):
        """Run diffusion-based action prediction"""
        # Clone embedding for reuse in each timestep
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise
        noise_preds = []
        timesteps = list(action_head.noise_scheduler.timesteps)
        grad_denoise_steps = max(0, int(grad_denoise_steps))
        collect_from = max(0, len(timesteps) - grad_denoise_steps)

        # Reverse diffusion: Iteratively denoise to generate action prediction
        for step_idx, t in enumerate(timesteps):
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
            # embedding, and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] Replace the embeddings of the action tokens with noisy actions
            # (Later on, the positional embeddings will be added to them)

            # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Reshape and project noisy actions into language embedding space
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Replace action token embeddings with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Build multimodal embeddings and attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Forward pass through language model
            if timing is not None:
                if timer_device is None:
                    timer_device = input_embeddings.device
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                lm_start = time.perf_counter()
            if profile_flops and flops is not None:
                flops["language"] = flops.get("language", 0.0) + self._estimate_llm_flops(
                    multimodal_embeddings.shape[1], multimodal_embeddings.shape[0]
                )
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if timing is not None:
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                timing["language"] = timing.get("language", 0.0) + (time.perf_counter() - lm_start)

            # Extract hidden states for action portion of response
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # Predict noise and update noisy actions: x_t -> x_{t-1}
            if timing is not None:
                if timer_device is None:
                    timer_device = input_embeddings.device
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                head_start = time.perf_counter()
            if profile_flops and flops is not None:
                num_chunks = actions_hidden_states.shape[1] // ACTION_DIM
                flops["action"] = flops.get("action", 0.0) + self._estimate_action_head_flops(
                    action_head, num_chunks, actions_hidden_states.shape[0]
                )
            noise_pred = action_head.predict_noise(actions_hidden_states)
            if timing is not None:
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                timing["action"] = timing.get("action", 0.0) + (time.perf_counter() - head_start)
            if collect_noise_preds and step_idx >= collect_from:
                noise_preds.append(noise_pred)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states.detach(), noise_preds

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        return_actions_tensor: bool = False,
        return_attentions: bool = False,
        timing: Optional[Dict[str, float]] = None,
        timer_device: Optional[torch.device] = None,
        flops: Optional[Dict[str, float]] = None,
        profile_flops: bool = False,
        profile_flops_silent: bool = False,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        if timing is not None:
            if timer_device is None:
                timer_device = input_embeddings.device
            if timer_device.type == "cuda":
                torch.cuda.synchronize(timer_device)
            lm_start = time.perf_counter()
        if profile_flops and flops is not None:
            flops["language"] = flops.get("language", 0.0) + self._estimate_llm_flops(
                multimodal_embeddings.shape[1], multimodal_embeddings.shape[0]
            )
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=return_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        if timing is not None:
            if timer_device.type == "cuda":
                torch.cuda.synchronize(timer_device)
            timing["language"] = timing.get("language", 0.0) + (time.perf_counter() - lm_start)

        last_attn = None
        if return_attentions and language_model_output.attentions:
            last_attn = language_model_output.attentions[-1]

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # Handle different prediction methods
        normalized_actions_tensor = None
        if action_head is not None:
            # L1 regression prediction
            if timing is not None:
                if timer_device is None:
                    timer_device = input_embeddings.device
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                head_start = time.perf_counter()
            if profile_flops and flops is not None:
                num_chunks = actions_hidden_states.shape[1] // ACTION_DIM
                flops["action"] = flops.get("action", 0.0) + self._estimate_action_head_flops(
                    action_head, num_chunks, actions_hidden_states.shape[0]
                )
            normalized_actions_tensor = action_head.predict_action(actions_hidden_states)
            if timing is not None:
                if timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                timing["action"] = timing.get("action", 0.0) + (time.perf_counter() - head_start)
            normalized_actions = normalized_actions_tensor.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        if not return_actions_tensor:
            normalized_actions_tensor = None
        if return_attentions:
            return normalized_actions, actions_hidden_states, normalized_actions_tensor, last_attn
        return normalized_actions, actions_hidden_states, normalized_actions_tensor

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        token_selection_cfg: Optional[TokenSelectionConfig] = None,
        return_token_selection: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            token_selection_cfg: Optional token selection configuration
            return_token_selection: If True, returns token selection state as well
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states) with optional token selection state
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")
        use_regression = action_head is not None and hasattr(action_head, "predict_action") and not use_diffusion

        token_cfg = self._resolve_token_selection_cfg(token_selection_cfg)
        token_selection_active = (
            token_cfg is not None
            and token_cfg.token_selection_enabled
            and (use_diffusion or use_regression)
        )
        profile_flops = bool(getattr(token_cfg, "flops_profile_enabled", False)) if token_cfg is not None else False
        profile_flops_silent = (
            bool(getattr(token_cfg, "flops_profile_silent", True)) if token_cfg is not None else True
        )
        flops: Optional[Dict[str, float]] = None
        if profile_flops:
            flops = {"vision": 0.0, "language": 0.0, "action": 0.0}
        if token_cfg is not None and token_cfg.token_selection_enabled and not (use_diffusion or use_regression):
            logger.warning(
                "Token selection is enabled but neither diffusion nor regression is active; skipping token selection."
            )

        if not token_selection_active:
            timing: Dict[str, float] = {"vision": 0.0, "language": 0.0, "action": 0.0}
            timer_device: Optional[torch.device] = None
            if torch.is_tensor(pixel_values):
                timer_device = pixel_values.device
            elif isinstance(pixel_values, dict):
                for value in pixel_values.values():
                    if torch.is_tensor(value):
                        timer_device = value.device
                        break

            def _timing_start():
                if timer_device is not None and timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                return time.perf_counter()

            def _timing_stop(start: float) -> float:
                if timer_device is not None and timer_device.type == "cuda":
                    torch.cuda.synchronize(timer_device)
                return time.perf_counter() - start

            # Prepare inputs by adding necessary tokens
            input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

            # Update labels tensor for action mask computation later
            labels = self._prepare_labels_for_action_prediction(labels, input_ids)

            # Get input embeddings and action masks
            input_embeddings = self.get_input_embeddings()(input_ids)
            all_actions_mask = self._process_action_masks(labels)

            # Extract language embeddings
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )

            # Process vision features
            vision_start = _timing_start()
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
            timing["vision"] = timing.get("vision", 0.0) + _timing_stop(vision_start)
            if profile_flops and flops is not None:
                flops["vision"] = flops.get("vision", 0.0) + self._estimate_vision_total_flops(
                    projected_patch_embeddings.shape[0]
                )

            # Add proprioceptive features if provided
            use_proprio = proprio_projector is not None and proprio is not None
            if use_proprio:
                proprio = torch.Tensor(proprio).to(
                    projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype
                )
                projected_patch_embeddings = self._process_proprio_features(
                    projected_patch_embeddings, proprio, proprio_projector
                )

            # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
            NUM_PATCHES = projected_patch_embeddings.shape[1]
            if use_diffusion:
                NUM_PATCHES += 1

            if use_diffusion:
                noise = torch.randn(
                    size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
                    device=input_embeddings.device,
                    dtype=input_embeddings.dtype,
                )
                normalized_actions, actions_hidden_states, _ = self._run_diffusion_prediction(
                    input_embeddings,
                    all_actions_mask,
                    noise,
                    action_head,
                    projected_patch_embeddings,
                    labels,
                    attention_mask,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    noisy_action_projector,
                    timing=timing,
                    timer_device=timer_device,
                    flops=flops,
                    profile_flops=profile_flops,
                    profile_flops_silent=profile_flops_silent,
                )
            else:
                normalized_actions, actions_hidden_states, _ = self._regression_or_discrete_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    action_head,
                    timing=timing,
                    timer_device=timer_device,
                    flops=flops,
                    profile_flops=profile_flops,
                    profile_flops_silent=profile_flops_silent,
                )

            actions = self._unnormalize_actions(normalized_actions, unnorm_key)
            if flops is not None:
                flops["total"] = float(flops.get("vision", 0.0) + flops.get("language", 0.0) + flops.get("action", 0.0))
            self._token_selection_state["last_timing"] = timing
            self._token_selection_state["last_flops"] = flops
            if return_token_selection:
                return actions, actions_hidden_states, None
            return actions, actions_hidden_states

        state = self._token_selection_state
        timing: Dict[str, float] = {"vision": 0.0, "language": 0.0, "action": 0.0, "eval": 0.0}
        timer_device: Optional[torch.device] = None
        if torch.is_tensor(pixel_values):
            timer_device = pixel_values.device
        elif isinstance(pixel_values, dict):
            for value in pixel_values.values():
                if torch.is_tensor(value):
                    timer_device = value.device
                    break

        def _timing_start():
            if timer_device is not None and timer_device.type == "cuda":
                torch.cuda.synchronize(timer_device)
            return time.perf_counter()

        def _timing_stop(start: float) -> float:
            if timer_device is not None and timer_device.type == "cuda":
                torch.cuda.synchronize(timer_device)
            return time.perf_counter() - start
        interval = max(1, int(token_cfg.region_eval_interval))
        eval_frame = (state["frame_idx"] % interval) == 0
        state["frame_idx"] += 1
        state["last_eval_frame"] = eval_frame
        score_method = str(getattr(token_cfg, "grad_score_method", "full_grad")).lower()
        use_partial_grad = eval_frame and score_method == "partial_grad"
        use_attn_only = eval_frame and score_method == "attn_only"
        if (use_partial_grad or use_attn_only) and use_diffusion:
            logger.warning("Attention/partial-grad scoring does not support diffusion; falling back to full grad.")
            use_partial_grad = False
            use_attn_only = False
        use_full_grad = eval_frame and not (use_partial_grad or use_attn_only)
        reuse_mode = str(getattr(token_cfg, "token_reuse_mode", "none")).lower()
        if reuse_mode not in {"none", "reuse_kv", "reuse_all"}:
            reuse_mode = "none"

        with torch.set_grad_enabled(use_full_grad):
            # Prepare inputs by adding necessary tokens
            input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

            # Update labels tensor for action mask computation later
            labels = self._prepare_labels_for_action_prediction(labels, input_ids)

            # Get input embeddings and action masks
            input_embeddings = self.get_input_embeddings()(input_ids)
            all_actions_mask = self._process_action_masks(labels)

            # Extract language embeddings
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )

            # Process vision features
            vision_start = _timing_start()
            image_embs_new = self._process_vision_features(pixel_values, language_embeddings, use_film)
            timing["vision"] = timing.get("vision", 0.0) + _timing_stop(vision_start)
            if profile_flops and flops is not None:
                flops["vision"] = flops.get("vision", 0.0) + self._estimate_vision_total_flops(
                    image_embs_new.shape[0]
                )
            image_embs = image_embs_new
            if (
                token_cfg.vision_partial_update_enabled
                and not eval_frame
                and state["last_image_embs"] is not None
            ):
                update_mask = state["last_effective_important_mask"]
                if update_mask is None:
                    update_mask = state["last_important_mask"]
                if update_mask is None:
                    update_mask = torch.ones(
                        image_embs_new.shape[:2], device=image_embs_new.device, dtype=torch.bool
                    )
                if update_mask.dim() == 1:
                    update_mask = update_mask.unsqueeze(0)
                image_embs = torch.where(update_mask.unsqueeze(-1), image_embs_new, state["last_image_embs"])

            if use_full_grad:
                image_embs.requires_grad_(True)

            background_region_mask = self._compute_background_region_mask(
                image_embs.detach(),
                state["last_image_embs"],
                token_cfg.token_temporal_threshold,
                token_cfg.token_spatial_threshold,
                token_cfg.token_spatial_radius,
                token_cfg.region_patch_size,
            )
            background_mask = self._expand_region_mask(background_region_mask, token_cfg.region_patch_size)

            if not eval_frame:
                important_region_mask = state.get("last_important_region_mask")
                if important_region_mask is None:
                    important_region_mask = torch.zeros_like(background_region_mask, dtype=torch.bool)
                token_scores = state["last_token_scores"]
                if token_scores is None:
                    token_scores = torch.linalg.norm(image_embs.detach().float(), dim=-1)
                    state["last_token_scores"] = token_scores.detach()
                region_scores = self._compute_region_scores(token_scores, token_cfg.region_patch_size)

                keep_pre_region = (~background_region_mask) | important_region_mask
                keep_region_final, clipped_region_mask = self._apply_keep_constraints_regions(
                    keep_pre_region,
                    region_scores,
                    token_cfg.min_kept_tokens,
                    token_cfg.max_kept_tokens,
                    token_cfg.region_patch_size,
                )
                important_mask = self._expand_region_mask(important_region_mask, token_cfg.region_patch_size)
                keep_pre = self._expand_region_mask(keep_pre_region, token_cfg.region_patch_size)
                keep_final = self._expand_region_mask(keep_region_final, token_cfg.region_patch_size)
                clipped_mask = self._expand_region_mask(clipped_region_mask, token_cfg.region_patch_size)
                effective_important_mask = important_mask & keep_final
                overlay = self._build_overlay_labels(keep_final, important_mask, clipped_mask)

                if token_cfg.token_prune_enabled and reuse_mode == "none":
                    if image_embs.shape[0] != 1:
                        raise ValueError("Token pruning only supports batch size 1.")
                    kept_indices = torch.where(keep_final[0])[0]
                    image_embs_for_llm = image_embs[:, kept_indices, :]
                else:
                    image_embs_for_llm = image_embs

                use_proprio = proprio_projector is not None and proprio is not None
                if use_proprio:
                    proprio = torch.Tensor(proprio).to(
                        image_embs_for_llm.device, dtype=image_embs_for_llm.dtype
                    )
                    projected_patch_embeddings = self._process_proprio_features(
                        image_embs_for_llm, proprio, proprio_projector
                    )
                else:
                    projected_patch_embeddings = image_embs_for_llm

                if use_diffusion:
                    NUM_PATCHES = projected_patch_embeddings.shape[1] + 1
                    noise = torch.randn(
                        size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
                        device=input_embeddings.device,
                        dtype=input_embeddings.dtype,
                    )
                    if reuse_mode != "none":
                        reuse_mask_full = self._build_llm_reuse_mask(
                            keep_final, projected_patch_embeddings.shape[1], input_embeddings
                        )
                        self._set_llm_reuse_context(reuse_mask_full, reuse_mode, force=True)
                    normalized_actions, actions_hidden_states, _ = self._run_diffusion_prediction(
                        input_embeddings,
                        all_actions_mask,
                        noise,
                        action_head,
                        projected_patch_embeddings,
                        labels,
                        attention_mask,
                        NUM_PATCHES,
                        NUM_PROMPT_TOKENS,
                        noisy_action_projector,
                        timing=timing,
                        timer_device=timer_device,
                        flops=flops,
                        profile_flops=profile_flops,
                        profile_flops_silent=profile_flops_silent,
                    )
                    if reuse_mode != "none":
                        self._clear_llm_reuse_context()
                else:
                    NUM_PATCHES = projected_patch_embeddings.shape[1]
                    if reuse_mode != "none":
                        reuse_mask_full = self._build_llm_reuse_mask(
                            keep_final, projected_patch_embeddings.shape[1], input_embeddings
                        )
                        self._set_llm_reuse_context(reuse_mask_full, reuse_mode, force=True)
                    normalized_actions, actions_hidden_states, _ = self._regression_or_discrete_prediction(
                        input_embeddings,
                        all_actions_mask,
                        projected_patch_embeddings,
                        attention_mask,
                        labels,
                        NUM_PATCHES,
                        NUM_PROMPT_TOKENS,
                        action_head,
                        timing=timing,
                        timer_device=timer_device,
                        flops=flops,
                        profile_flops=profile_flops,
                        profile_flops_silent=profile_flops_silent,
                    )
                    if reuse_mode != "none":
                        self._clear_llm_reuse_context()

            else:
                use_proprio = proprio_projector is not None and proprio is not None
                if use_proprio:
                    proprio = torch.Tensor(proprio).to(image_embs.device, dtype=image_embs.dtype)
                    projected_patch_embeddings = self._process_proprio_features(
                        image_embs, proprio, proprio_projector
                    )
                else:
                    projected_patch_embeddings = image_embs

                normalized_actions_tensor = None
                noise_preds = None
                last_attn = None
                if use_diffusion:
                    NUM_PATCHES = projected_patch_embeddings.shape[1] + 1
                    noise = torch.randn(
                        size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
                        device=input_embeddings.device,
                        dtype=input_embeddings.dtype,
                    )
                    if reuse_mode != "none":
                        self._set_llm_reuse_context(None, reuse_mode, force=True)
                    normalized_actions, actions_hidden_states, noise_preds = self._run_diffusion_prediction(
                        input_embeddings,
                        all_actions_mask,
                        noise,
                        action_head,
                        projected_patch_embeddings,
                        labels,
                        attention_mask,
                        NUM_PATCHES,
                        NUM_PROMPT_TOKENS,
                        noisy_action_projector,
                        collect_noise_preds=True,
                        grad_denoise_steps=token_cfg.grad_denoise_steps,
                        timing=timing,
                        timer_device=timer_device,
                        flops=flops,
                        profile_flops=profile_flops,
                        profile_flops_silent=profile_flops_silent,
                    )
                    if reuse_mode != "none":
                        self._clear_llm_reuse_context()
                else:
                    NUM_PATCHES = projected_patch_embeddings.shape[1]
                    if use_partial_grad:
                        if reuse_mode != "none":
                            self._set_llm_reuse_context(None, reuse_mode, force=True)
                        (
                            normalized_actions,
                            actions_hidden_states,
                            normalized_actions_tensor,
                            last_attn,
                        ) = self._regression_or_discrete_prediction(
                            input_embeddings,
                            all_actions_mask,
                            projected_patch_embeddings,
                            attention_mask,
                            labels,
                            NUM_PATCHES,
                            NUM_PROMPT_TOKENS,
                            action_head,
                            return_actions_tensor=True,
                            return_attentions=True,
                            timing=timing,
                            timer_device=timer_device,
                            flops=flops,
                            profile_flops=profile_flops,
                            profile_flops_silent=profile_flops_silent,
                        )
                        if reuse_mode != "none":
                            self._clear_llm_reuse_context()
                    elif use_attn_only:
                        if reuse_mode != "none":
                            self._set_llm_reuse_context(None, reuse_mode, force=True)
                        (
                            normalized_actions,
                            actions_hidden_states,
                            normalized_actions_tensor,
                            last_attn,
                        ) = self._regression_or_discrete_prediction(
                            input_embeddings,
                            all_actions_mask,
                            projected_patch_embeddings,
                            attention_mask,
                            labels,
                            NUM_PATCHES,
                            NUM_PROMPT_TOKENS,
                            action_head,
                            return_actions_tensor=True,
                            return_attentions=True,
                            timing=timing,
                            timer_device=timer_device,
                            flops=flops,
                            profile_flops=profile_flops,
                            profile_flops_silent=profile_flops_silent,
                        )
                        if reuse_mode != "none":
                            self._clear_llm_reuse_context()
                    else:
                        if reuse_mode != "none":
                            self._set_llm_reuse_context(None, reuse_mode, force=True)
                        normalized_actions, actions_hidden_states, normalized_actions_tensor = (
                            self._regression_or_discrete_prediction(
                                input_embeddings,
                                all_actions_mask,
                                projected_patch_embeddings,
                                attention_mask,
                                labels,
                                NUM_PATCHES,
                                NUM_PROMPT_TOKENS,
                                action_head,
                                return_actions_tensor=True,
                                timing=timing,
                                timer_device=timer_device,
                                flops=flops,
                                profile_flops=profile_flops,
                                profile_flops_silent=profile_flops_silent,
                            )
                        )
                        if reuse_mode != "none":
                            self._clear_llm_reuse_context()
                    if normalized_actions_tensor is None:
                        raise ValueError("Token selection requires regression action head when diffusion is disabled.")

                eval_start = _timing_start()
                token_scores = None
                gate = 0.0
                lambda_pos = 1.0
                lambda_grip = 1.0
                if use_partial_grad:
                    lambda_pos = float(token_cfg.partial_grad_pos_weight)
                    lambda_grip = float(token_cfg.partial_grad_grip_weight)
                    token_scores = self._compute_partial_grad_scores(
                        action_head,
                        actions_hidden_states,
                        normalized_actions_tensor,
                        last_attn,
                        NUM_PROMPT_TOKENS,
                        NUM_PATCHES,
                        token_cfg,
                    )
                    if token_scores is None:
                        token_scores = torch.linalg.norm(image_embs.detach().float(), dim=-1)
                elif use_attn_only:
                    token_scores = self._compute_attn_only_scores(
                        last_attn,
                        actions_hidden_states.shape[1],
                        NUM_PROMPT_TOKENS,
                        NUM_PATCHES,
                        token_cfg,
                    )
                    if token_scores is None:
                        token_scores = torch.linalg.norm(image_embs.detach().float(), dim=-1)
                else:
                    if token_cfg.grad_tau > 0:
                        gripper = normalized_actions[:, -1]
                        if gripper.shape[0] > 1:
                            m = float(np.mean(np.abs(gripper[1:] - gripper[:-1])))
                        else:
                            m = 0.0
                        gate = float(np.clip(m / token_cfg.grad_tau, 0.0, 1.0))
                    lambda_grip = 1.0 + token_cfg.grad_alpha * gate
                    lambda_pos = 1.0 + token_cfg.grad_beta * (1.0 - gate)

                    objective = torch.zeros((), device=image_embs.device, dtype=image_embs.dtype)
                    if use_diffusion:
                        for noise_pred in noise_preds:
                            v_pos = noise_pred[..., :-1]
                            v_grip = noise_pred[..., -1]
                            objective = objective + lambda_pos * (v_pos.pow(2).sum()) + lambda_grip * (
                                v_grip.pow(2).sum()
                            )
                    else:
                        v_pos = normalized_actions_tensor[..., :-1]
                        v_grip = normalized_actions_tensor[..., -1]
                        objective = objective + lambda_pos * (v_pos.pow(2).sum()) + lambda_grip * (
                            v_grip.pow(2).sum()
                        )

                    image_grads = torch.autograd.grad(
                        objective,
                        image_embs,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if image_grads is None:
                        image_grads = torch.zeros_like(image_embs)

                    # Use float32 for norms to avoid unsupported bf16 linalg ops.
                    grad_norm = torch.linalg.norm(image_grads.float(), dim=-1)
                    token_norm = torch.linalg.norm(image_embs.detach().float(), dim=-1)
                    token_scores = grad_norm * token_norm
                    token_scores = token_scores.detach()

                if token_scores is not None:
                    token_scores = token_scores.detach()

                region_scores = self._compute_region_scores(token_scores, token_cfg.region_patch_size)
                if token_cfg.grad_region_ema is not None and state["last_region_scores"] is not None:
                    if state["last_region_scores"].shape == region_scores.shape:
                        region_scores = token_cfg.grad_region_ema * state["last_region_scores"] + (
                            1.0 - token_cfg.grad_region_ema
                        ) * region_scores

                mass = float(min(max(token_cfg.grad_region_mass, 0.0), 1.0))
                important_region_mask = self._select_important_mask(
                    region_scores, mass, token_cfg.region_patch_size, return_region_mask=True
                )
                if token_cfg.grad_keep_prev and state.get("last_important_region_mask") is not None:
                    important_region_mask = important_region_mask | state["last_important_region_mask"]

                keep_pre_region = (~background_region_mask) | important_region_mask
                keep_region_final, clipped_region_mask = self._apply_keep_constraints_regions(
                    keep_pre_region,
                    region_scores,
                    token_cfg.min_kept_tokens,
                    token_cfg.max_kept_tokens,
                    token_cfg.region_patch_size,
                )
                important_mask = self._expand_region_mask(important_region_mask, token_cfg.region_patch_size)
                keep_pre = self._expand_region_mask(keep_pre_region, token_cfg.region_patch_size)
                keep_final = self._expand_region_mask(keep_region_final, token_cfg.region_patch_size)
                clipped_mask = self._expand_region_mask(clipped_region_mask, token_cfg.region_patch_size)
                effective_important_mask = important_mask & keep_final
                overlay = self._build_overlay_labels(keep_final, important_mask, clipped_mask)
                timing["eval"] = timing.get("eval", 0.0) + _timing_stop(eval_start)

                state["last_token_scores"] = token_scores.detach()
                state["last_region_scores"] = region_scores.detach()
                state["last_important_mask"] = important_mask.detach()
                state["last_important_region_mask"] = important_region_mask.detach()
                state["last_gate"] = gate
                state["last_lambda_pos"] = lambda_pos
                state["last_lambda_grip"] = lambda_grip

            state["last_image_embs"] = image_embs.detach()
            state["last_background_mask"] = background_mask.detach()
            state["last_keep_pre_mask"] = keep_pre.detach()
            state["last_keep_mask"] = keep_final.detach()
            state["last_clipped_mask"] = clipped_mask.detach()
            state["last_effective_important_mask"] = effective_important_mask.detach()
            state["last_overlay_labels"] = overlay.detach()
            state["last_timing"] = timing
            if flops is not None:
                flops["total"] = float(
                    flops.get("vision", 0.0) + flops.get("language", 0.0) + flops.get("action", 0.0)
                )
            state["last_flops"] = flops

            num_images = self.vision_backbone.get_num_images_in_input()
            num_patches = self.vision_backbone.get_num_patches()
            grid_h, grid_w = self._get_patch_grid_size(num_patches)
            state["last_overlay_grid"] = self._reshape_tokens_to_grid(
                overlay.detach(), num_images, num_patches, grid_h, grid_w
            )

        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        if return_token_selection:
            return actions, actions_hidden_states, state
        return actions, actions_hidden_states

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
