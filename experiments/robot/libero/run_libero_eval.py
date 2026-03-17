"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import shlex
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../LIBERO'))
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Token selection & pruning (diffusion/regression inference)
    #################################################################################################################
    token_selection_enabled: bool = True             # Enable gradient-based token selection
    token_prune_enabled: bool = False                # Enable LLM token pruning on non-eval frames
    vision_partial_update_enabled: bool = False      # Enable partial vision token updates (cache reuse)
    flops_profile_enabled: bool = False              # Enable FLOPs estimate (theoretical, per-forward)
    flops_profile_silent: bool = True                # No-op for estimate mode
    region_eval_interval: int = 1                    # Evaluate token importance every N frames
    grad_denoise_steps: int = 1                      # Use last N denoise steps for gradient objective
    grad_region_mass: float = 0.70                   # Mass threshold for region selection (normalized)
    grad_region_ema: Optional[float] = 0.70          # EMA factor for region scores (None disables)
    grad_keep_prev: bool = False                     # Union important mask with previous frame
    grad_score_method: str = "full_grad"             # "full_grad", "partial_grad", or "attn_only"
    partial_grad_phi: str = "l2"                     # "l2" or "l1"
    partial_grad_pos_weight: float = 1.0             # Pos dims weight for partial grad
    partial_grad_grip_weight: float = 2.0            # Grip dim weight for partial grad
    attn_score_beta: float = 1.0                     # Exponent for attention-based scoring (beta > 0)
    head_consensus_gating_enabled: bool = False      # Enable head-consensus gating in partial-grad scoring
    head_consensus_eta: float = 1.0                  # Exponent for head-consensus gating (eta > 0)
    head_g_beta: float = 1.0                         # Exponent for head magnitude in partial-grad (beta > 0)
    head_g_norm: str = "sum"                         # Head g normalization across heads: sum or max
    token_reuse_mode: str = "none"                   # "none", "reuse_kv", or "reuse_all"
    grad_tau: float = 0.1                            # Gripper change scale for adaptive weighting
    grad_alpha: float = 1.0                          # Gripper weight scale
    grad_beta: float = 1.0                           # Position weight scale
    token_temporal_threshold: float = 0.95           # Temporal cosine similarity threshold
    token_spatial_threshold: float = 0.9             # Spatial cosine similarity threshold
    token_spatial_radius: int = 1                    # Spatial neighbor radius (in tokens)
    min_kept_tokens: int = 1                         # Minimum tokens to keep
    max_kept_tokens: Optional[int] = None            # Maximum tokens to keep (None disables)
    region_patch_size: int = 1                       # Region size in patch tokens

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    rollout_dir: str = "./rollouts"                  # Base directory for rollout MP4s
    overlay_show_scores: bool = True                 # Whether to draw region scores on overlays
    overlay_show_ids: bool = False                   # Whether to draw region IDs on overlays

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.token_selection_enabled:
        assert (cfg.use_diffusion or cfg.use_l1_regression), (
            "Token selection requires diffusion or L1 regression action prediction."
        )
        assert cfg.region_eval_interval >= 1, "region_eval_interval must be >= 1."
        assert cfg.grad_denoise_steps >= 1, "grad_denoise_steps must be >= 1."
        if cfg.max_kept_tokens is not None:
            assert cfg.max_kept_tokens >= cfg.min_kept_tokens, "max_kept_tokens must be >= min_kept_tokens."
        score_method = str(cfg.grad_score_method).lower()
        assert score_method in {"full_grad", "partial_grad", "attn_only"}, (
            "grad_score_method must be one of: full_grad, partial_grad, attn_only."
        )
        if score_method in {"partial_grad", "attn_only"}:
            assert cfg.use_l1_regression and not cfg.use_diffusion, (
                "partial_grad/attn_only scoring requires L1 regression and does not support diffusion."
            )
        if score_method == "partial_grad":
            phi = str(cfg.partial_grad_phi).lower()
            assert phi in {"l1", "l2"}, "partial_grad_phi must be 'l1' or 'l2'."
        attn_beta = float(getattr(cfg, "attn_score_beta", 1.0))
        assert attn_beta > 0.0, "attn_score_beta must be > 0."
        head_consensus = bool(getattr(cfg, "head_consensus_gating_enabled", False))
        if head_consensus:
            assert score_method == "partial_grad", "head-consensus gating requires partial_grad scoring."
            head_eta = float(getattr(cfg, "head_consensus_eta", 1.0))
            assert head_eta > 0.0, "head_consensus_eta must be > 0."
        head_g_beta = float(getattr(cfg, "head_g_beta", 1.0))
        assert head_g_beta > 0.0, "head_g_beta must be > 0."
        if score_method == "partial_grad":
            head_g_norm = str(getattr(cfg, "head_g_norm", "sum")).lower()
            assert head_g_norm in {"sum", "max"}, "head_g_norm must be 'sum' or 'max'."
        reuse_mode = str(cfg.token_reuse_mode).lower()
        assert reuse_mode in {"none", "reuse_kv", "reuse_all"}, (
            "token_reuse_mode must be one of: none, reuse_kv, reuse_all."
        )
        if reuse_mode != "none":
            assert cfg.use_l1_regression and not cfg.use_diffusion, (
                "token_reuse_mode requires L1 regression and does not support diffusion."
            )


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def _extract_overlay_state(model, cfg):
    if not hasattr(model, "get_token_selection_state"):
        return None, None

    state = model.get_token_selection_state()
    overlay_grid = state.get("last_overlay_grid")
    if overlay_grid is None:
        return None, None

    show_scores = bool(getattr(cfg, "overlay_show_scores", True))
    show_ids = bool(getattr(cfg, "overlay_show_ids", False))

    if torch.is_tensor(overlay_grid):
        overlay_grid = overlay_grid.detach().cpu().numpy()
    overlay_grid = overlay_grid.astype(np.uint8)

    if overlay_grid.ndim == 4:
        overlay_grid = overlay_grid[0, 0]
    elif overlay_grid.ndim == 3:
        overlay_grid = overlay_grid[0]

    region_patch = max(1, int(cfg.region_patch_size))
    token_scores = state.get("last_token_scores")
    if token_scores is not None:
        if torch.is_tensor(token_scores):
            token_scores = token_scores.detach().cpu().numpy()
        if token_scores.ndim == 2:
            token_scores = token_scores[0]
        tokens_per_image = overlay_grid.shape[0] * overlay_grid.shape[1]
        token_scores = token_scores[:tokens_per_image]
        token_scores = token_scores.reshape(overlay_grid.shape)

    region_scores = None
    if show_scores:
        region_scores = state.get("last_region_scores")
        if region_scores is not None:
            if torch.is_tensor(region_scores):
                region_scores = region_scores.detach().cpu().numpy()
            if region_scores.ndim == 4:
                region_scores = region_scores[0, 0]
            elif region_scores.ndim == 3:
                region_scores = region_scores[0]
        elif token_scores is not None and region_patch > 1:
            if (
                overlay_grid.shape[0] % region_patch == 0
                and overlay_grid.shape[1] % region_patch == 0
            ):
                region_h = overlay_grid.shape[0] // region_patch
                region_w = overlay_grid.shape[1] // region_patch
                region_scores = token_scores.reshape(
                    region_h, region_patch, region_w, region_patch
                ).mean(axis=(1, 3))

    if region_patch > 1:
        if overlay_grid.shape[0] % region_patch == 0 and overlay_grid.shape[1] % region_patch == 0:
            region_h = overlay_grid.shape[0] // region_patch
            region_w = overlay_grid.shape[1] // region_patch
            labels = overlay_grid.reshape(region_h, region_patch, region_w, region_patch)
            if token_scores is not None:
                scores = token_scores.reshape(region_h, region_patch, region_w, region_patch)
                label_scores = np.zeros((4, region_h, region_w), dtype=np.float32)
                for label in range(4):
                    label_scores[label] = (scores * (labels == label)).sum(axis=(1, 3))
                region_labels = label_scores.argmax(axis=0).astype(np.uint8)
            else:
                label_counts = np.zeros((4, region_h, region_w), dtype=np.int32)
                for label in range(4):
                    label_counts[label] = (labels == label).sum(axis=(1, 3))
                region_labels = label_counts.argmax(axis=0).astype(np.uint8)
            overlay_grid = region_labels
            if region_scores is not None and region_scores.shape != overlay_grid.shape and token_scores is not None:
                region_scores = token_scores.reshape(
                    region_h, region_patch, region_w, region_patch
                ).mean(axis=(1, 3))

    return overlay_grid, region_scores


def _extract_region_scores_for_plot(model, cfg):
    if not hasattr(model, "get_token_selection_state"):
        return None

    state = model.get_token_selection_state()
    if state is None:
        return None

    def _to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.array(value)

    overlay_grid = _to_numpy(state.get("last_overlay_grid"))
    if overlay_grid is not None:
        if overlay_grid.ndim == 4:
            overlay_grid = overlay_grid[0, 0]
        elif overlay_grid.ndim == 3:
            overlay_grid = overlay_grid[0]

    grid_h = overlay_grid.shape[0] if overlay_grid is not None else None
    grid_w = overlay_grid.shape[1] if overlay_grid is not None else None
    region_patch = max(1, int(cfg.region_patch_size))

    region_scores = _to_numpy(state.get("last_region_scores"))
    if region_scores is not None:
        if region_scores.ndim == 4:
            region_scores = region_scores[0, 0]
        elif region_scores.ndim == 3:
            region_scores = region_scores[0]
        elif region_scores.ndim == 2 and region_scores.shape[0] == 1:
            region_scores = region_scores[0]

        if region_scores.ndim == 1 and grid_h is not None and grid_w is not None:
            if (grid_h % region_patch) != 0 or (grid_w % region_patch) != 0:
                return None
            region_h = grid_h // region_patch
            region_w = grid_w // region_patch
            if region_scores.size == region_h * region_w:
                region_scores = region_scores.reshape(region_h, region_w)

        if region_scores.ndim == 2:
            return region_scores.astype(np.float32)

    token_scores = _to_numpy(state.get("last_token_scores"))
    if token_scores is None or overlay_grid is None:
        return None

    if token_scores.ndim == 2:
        token_scores = token_scores[0]
    tokens_per_image = grid_h * grid_w
    if tokens_per_image <= 0:
        return None
    if token_scores.size < tokens_per_image:
        return None
    token_scores = token_scores[:tokens_per_image]
    scores_grid = token_scores.reshape(grid_h, grid_w)
    if region_patch == 1:
        return scores_grid.astype(np.float32)
    if (grid_h % region_patch) != 0 or (grid_w % region_patch) != 0:
        return None
    region_h = grid_h // region_patch
    region_w = grid_w // region_patch
    region_scores = scores_grid.reshape(region_h, region_patch, region_w, region_patch).mean(axis=(1, 3))
    return region_scores.astype(np.float32)


def _apply_overlay(img, overlay_grid, region_scores=None, alpha=0.35, show_ids=False):
    if overlay_grid is None:
        return img

    img_np = img.astype(np.uint8, copy=False)
    height, width = img_np.shape[:2]

    overlay_img = Image.fromarray(overlay_grid).resize((width, height), resample=Image.NEAREST)
    overlay_labels = np.array(overlay_img)

    overlay_colors = np.zeros_like(img_np)
    overlay_colors[overlay_labels == 0] = (0, 255, 0)
    overlay_colors[overlay_labels == 1] = (255, 255, 0)
    overlay_colors[overlay_labels == 2] = (255, 0, 0)
    overlay_colors[overlay_labels == 3] = (0, 0, 255)

    blended = (img_np.astype(np.float32) * (1.0 - alpha) + overlay_colors.astype(np.float32) * alpha).astype(
        np.uint8
    )

    show_scores = region_scores is not None
    if not show_scores and not show_ids:
        return blended

    if torch.is_tensor(region_scores):
        region_scores = region_scores.detach().cpu().numpy()

    region_h, region_w = overlay_grid.shape
    cell_w = width / max(region_w, 1)
    cell_h = height / max(region_h, 1)

    if show_scores:
        region_scores = region_scores.astype(np.float32)
        total = float(region_scores.sum())
        if total > 0:
            region_scores = region_scores / total

    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    id_font_size = max(10, min(16, int(min(cell_w, cell_h) * 0.55)))
    score_font_size = max(7, min(12, int(min(cell_w, cell_h) * 0.38)))

    def _load_font(size):
        for font_name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    id_font = _load_font(id_font_size)
    score_font = _load_font(score_font_size)

    for r in range(region_h):
        for c in range(region_w):
            x = int((c + 0.03) * cell_w)
            y = int((r + 0.03) * cell_h)
            if show_ids:
                region_id = r * region_w + c
                id_text = str(region_id)
                draw.text((x + 1, y + 1), id_text, fill=(0, 0, 0), font=id_font)
                draw.text((x, y), id_text, fill=(255, 255, 255), font=id_font)
                y = y + int(id_font_size * 0.85)
            if show_scores:
                score = region_scores[r, c]
                text = f"{score:.3f}"
                draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=score_font)
                draw.text((x, y), text, fill=(255, 255, 255), font=score_font)

    return np.array(pil_img)


def save_region_score_plot(region_scores_history, mp4_path, log_file=None):
    if not region_scores_history:
        log_message("Region score plot skipped (no region scores collected).", log_file)
        return None

    scores = np.stack(region_scores_history, axis=0)
    if scores.ndim != 3:
        log_message("Region score plot skipped (unexpected score shape).", log_file)
        return None

    num_steps, region_h, region_w = scores.shape
    num_regions = region_h * region_w
    scores_flat = scores.reshape(num_steps, num_regions)

    row_sums = scores_flat.sum(axis=1, keepdims=True)
    nonzero = row_sums > 0
    scores_flat = np.where(nonzero, scores_flat / row_sums, scores_flat)

    fig_w = max(14.0, min(60.0, 16.0 + num_regions / 18.0))
    fig_h = max(8.0, min(36.0, 8.0 + num_regions / 40.0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    x = np.arange(1, num_steps + 1)
    colors = plt.cm.viridis(np.linspace(0.0, 0.8, num_regions))
    y_max = 0.15
    for idx in range(num_regions):
        plt.plot(x, scores_flat[:, idx], color=colors[idx], linewidth=0.6, alpha=0.7)
        label_y = min(float(scores_flat[-1, idx]), y_max)
        plt.text(
            num_steps + 0.5,
            label_y,
            str(idx),
            color=colors[idx],
            fontsize=8,
            alpha=0.85,
            va="center",
        )
    plt.xlabel("forward_step")
    plt.ylabel("score")
    plt.title(f"Region score trajectories (regions={num_regions}, id=row-major)")
    plt.ylim(0.0, y_max)
    plt.xlim(1, num_steps + 1.5)
    plt.tight_layout()

    plot_path = str(mp4_path).replace(".mp4", "--region_scores.png")
    plt.savefig(plot_path)
    plt.close()
    log_message(f"Saved region score plot at path {plot_path}", log_file)
    return plot_path


def _init_forward_stats():
    return {
        "total_calls": 0,
        "total_time": 0.0,
        "total_steps": 0,
        "base": {
            "count": 0,
            "total": 0.0,
            "vision": 0.0,
            "language": 0.0,
            "action": 0.0,
            "flops_total": 0.0,
            "flops_vision": 0.0,
            "flops_language": 0.0,
            "flops_action": 0.0,
        },
        "eval": {
            "count": 0,
            "total": 0.0,
            "vision": 0.0,
            "language": 0.0,
            "action": 0.0,
            "eval": 0.0,
            "important_ratio": 0.0,
            "ratio_count": 0,
            "flops_total": 0.0,
            "flops_vision": 0.0,
            "flops_language": 0.0,
            "flops_action": 0.0,
        },
        "prune": {
            "count": 0,
            "total": 0.0,
            "vision": 0.0,
            "language": 0.0,
            "action": 0.0,
            "prune_ratio": 0.0,
            "important_ratio": 0.0,
            "background_ratio": 0.0,
            "ratio_count": 0,
            "flops_total": 0.0,
            "flops_vision": 0.0,
            "flops_language": 0.0,
            "flops_action": 0.0,
        },
    }


def _mask_ratio(mask):
    if mask is None:
        return None
    if torch.is_tensor(mask):
        mask = mask.detach()
    if mask.numel() == 0:
        return None
    return float(mask.float().mean().item())


def _update_forward_stats(stats, model, cfg, total_time):
    stats["total_calls"] += 1
    stats["total_time"] += total_time
    if not hasattr(model, "get_token_selection_state"):
        return
    state = model.get_token_selection_state()
    if not isinstance(state, dict):
        return
    timing = state.get("last_timing") or {}
    flops = state.get("last_flops") if isinstance(state.get("last_flops"), dict) else None
    if not cfg.token_selection_enabled:
        base = stats["base"]
        base["count"] += 1
        base["total"] += total_time
        base["vision"] += float(timing.get("vision", 0.0))
        base["language"] += float(timing.get("language", 0.0))
        base["action"] += float(timing.get("action", 0.0))
        if flops is not None:
            base["flops_total"] += float(flops.get("total", 0.0))
            base["flops_vision"] += float(flops.get("vision", 0.0))
            base["flops_language"] += float(flops.get("language", 0.0))
            base["flops_action"] += float(flops.get("action", 0.0))
        return

    eval_frame = bool(state.get("last_eval_frame", False))
    bucket = stats["eval"] if eval_frame else stats["prune"]
    bucket["count"] += 1
    bucket["total"] += total_time
    bucket["vision"] += float(timing.get("vision", 0.0))
    bucket["language"] += float(timing.get("language", 0.0))
    bucket["action"] += float(timing.get("action", 0.0))
    if flops is not None:
        bucket["flops_total"] += float(flops.get("total", 0.0))
        bucket["flops_vision"] += float(flops.get("vision", 0.0))
        bucket["flops_language"] += float(flops.get("language", 0.0))
        bucket["flops_action"] += float(flops.get("action", 0.0))
    if eval_frame:
        bucket["eval"] += float(timing.get("eval", 0.0))

    keep_mask = state.get("last_keep_mask")
    prune_ratio = None
    if keep_mask is not None:
        keep_ratio = _mask_ratio(keep_mask)
        if keep_ratio is not None:
            prune_ratio = 1.0 - keep_ratio

    important_mask = state.get("last_effective_important_mask")
    if important_mask is None:
        base_important = state.get("last_important_mask")
        if base_important is not None and keep_mask is not None:
            important_mask = base_important & keep_mask
    important_ratio = _mask_ratio(important_mask)
    background_ratio = _mask_ratio(state.get("last_background_mask")) if not eval_frame else None

    if prune_ratio is not None and not eval_frame:
        bucket["prune_ratio"] += prune_ratio
    if important_ratio is not None:
        bucket["important_ratio"] += important_ratio
    if background_ratio is not None:
        bucket["background_ratio"] += background_ratio
    if important_ratio is not None or prune_ratio is not None:
        bucket["ratio_count"] += 1


def _format_forward_stats(stats):
    if stats["total_calls"] == 0:
        return []
    lines = []
    total_avg = stats["total_time"] / stats["total_calls"]
    lines.append(
        f"Forward均值: {total_avg * 1000:.1f} ms | 全部运行数={stats['total_calls']} | 总步数={stats['total_steps']}"
    )

    def _fmt_ms(value):
        return f"{value * 1000:6.1f} ms"

    def _fmt_pct(value):
        return f"{value * 100:5.1f}%"

    def _fmt_tflops(value):
        return f"{value / 1e12:6.3f} T"

    def _pad(label, value, width):
        return f"{label} {value}".ljust(width)

    base_stats = stats["base"]
    if base_stats["count"] > 0:
        lines.append(
            "推理均值: "
            + " | ".join(
                [
                    _pad("总", _fmt_ms(base_stats["total"] / base_stats["count"]), 10),
                    _pad("视觉", _fmt_ms(base_stats["vision"] / base_stats["count"]), 10),
                    _pad("语言", _fmt_ms(base_stats["language"] / base_stats["count"]), 10),
                    _pad("L1头", _fmt_ms(base_stats["action"] / base_stats["count"]), 10),
                    f"运行数={base_stats['count']}",
                ]
            )
        )
        if base_stats.get("flops_total", 0.0) > 0.0:
            lines.append(
                "总体FLOPS均值: "
                + " | ".join(
                    [
                        _pad("总", _fmt_tflops(base_stats["flops_total"] / base_stats["count"]), 10),
                        _pad("视觉", _fmt_tflops(base_stats["flops_vision"] / base_stats["count"]), 10),
                        _pad("语言", _fmt_tflops(base_stats["flops_language"] / base_stats["count"]), 10),
                        _pad("L1头", _fmt_tflops(base_stats["flops_action"] / base_stats["count"]), 10),
                        f"运行数={base_stats['count']}",
                    ]
                )
            )

    eval_stats = stats["eval"]
    if eval_stats["count"] > 0:
        ratio_count = max(1, eval_stats["ratio_count"])
        seg_total = _pad("总", _fmt_ms(eval_stats["total"] / eval_stats["count"]), 10)
        seg_important = _pad("重要", _fmt_pct(eval_stats["important_ratio"] / ratio_count), 10)
        seg_eval = _pad("评估", _fmt_ms(eval_stats["eval"] / eval_stats["count"]), 10)
        seg_vision = _pad("视觉", _fmt_ms(eval_stats["vision"] / eval_stats["count"]), 10)
        seg_language = _pad("语言", _fmt_ms(eval_stats["language"] / eval_stats["count"]), 10)
        seg_action = _pad("L1头", _fmt_ms(eval_stats["action"] / eval_stats["count"]), 10)
        seg_prune = _pad("剪枝", "--", 10)
        seg_background = _pad("背景", "--", 10)
        lines.append(
            "评估帧均值: "
            + " | ".join(
                [
                    seg_total,
                    seg_important,
                    seg_eval,
                    seg_vision,
                    seg_language,
                    seg_action,
                    seg_prune,
                    seg_background,
                    f"评估运行数={eval_stats['count']}",
                ]
            )
        )
        if eval_stats.get("flops_total", 0.0) > 0.0:
            lines.append(
                "评估FLOPS均值: "
                + " | ".join(
                    [
                        _pad("总", _fmt_tflops(eval_stats["flops_total"] / eval_stats["count"]), 10),
                        _pad("视觉", _fmt_tflops(eval_stats["flops_vision"] / eval_stats["count"]), 10),
                        _pad("语言", _fmt_tflops(eval_stats["flops_language"] / eval_stats["count"]), 10),
                        _pad("L1头", _fmt_tflops(eval_stats["flops_action"] / eval_stats["count"]), 10),
                        f"评估运行数={eval_stats['count']}",
                    ]
                )
            )

    prune_stats = stats["prune"]
    if prune_stats["count"] > 0:
        ratio_count = max(1, prune_stats["ratio_count"])
        seg_total = _pad("总", _fmt_ms(prune_stats["total"] / prune_stats["count"]), 10)
        seg_important = _pad("重要", _fmt_pct(prune_stats["important_ratio"] / ratio_count), 10)
        seg_eval = _pad("评估", "--", 10)
        seg_vision = _pad("视觉", _fmt_ms(prune_stats["vision"] / prune_stats["count"]), 10)
        seg_language = _pad("语言", _fmt_ms(prune_stats["language"] / prune_stats["count"]), 10)
        seg_action = _pad("L1头", _fmt_ms(prune_stats["action"] / prune_stats["count"]), 10)
        seg_prune = _pad("剪枝", _fmt_pct(prune_stats["prune_ratio"] / ratio_count), 10)
        seg_background = _pad("背景", _fmt_pct(prune_stats["background_ratio"] / ratio_count), 10)
        lines.append(
            "剪枝帧均值: "
            + " | ".join(
                [
                    seg_total,
                    seg_important,
                    seg_eval,
                    seg_vision,
                    seg_language,
                    seg_action,
                    seg_prune,
                    seg_background,
                    f"剪枝运行数={prune_stats['count']}",
                ]
            )
        )
        if prune_stats.get("flops_total", 0.0) > 0.0:
            lines.append(
                "剪枝FLOPS均值: "
                + " | ".join(
                    [
                        _pad("总", _fmt_tflops(prune_stats["flops_total"] / prune_stats["count"]), 10),
                        _pad("视觉", _fmt_tflops(prune_stats["flops_vision"] / prune_stats["count"]), 10),
                        _pad("语言", _fmt_tflops(prune_stats["flops_language"] / prune_stats["count"]), 10),
                        _pad("L1头", _fmt_tflops(prune_stats["flops_action"] / prune_stats["count"]), 10),
                        f"剪枝运行数={prune_stats['count']}",
                    ]
                )
            )

    if base_stats["count"] == 0:
        total_count = eval_stats["count"] + prune_stats["count"]
        total_flops = eval_stats.get("flops_total", 0.0) + prune_stats.get("flops_total", 0.0)
        if total_count > 0 and total_flops > 0.0:
            lines.append(
                "总体FLOPS均值: "
                + " | ".join(
                    [
                        _pad("总", _fmt_tflops(total_flops / total_count), 10),
                        _pad(
                            "视觉",
                            _fmt_tflops(
                                (eval_stats.get("flops_vision", 0.0) + prune_stats.get("flops_vision", 0.0))
                                / total_count
                            ),
                            10,
                        ),
                        _pad(
                            "语言",
                            _fmt_tflops(
                                (eval_stats.get("flops_language", 0.0) + prune_stats.get("flops_language", 0.0))
                                / total_count
                            ),
                            10,
                        ),
                        _pad(
                            "L1头",
                            _fmt_tflops(
                                (eval_stats.get("flops_action", 0.0) + prune_stats.get("flops_action", 0.0))
                                / total_count
                            ),
                            10,
                        ),
                        f"运行数={total_count}",
                    ]
                )
            )
    return lines


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()
    if cfg.token_selection_enabled and hasattr(model, "reset_token_selection_state"):
        model.reset_token_selection_state()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    last_overlay_grid = None
    last_region_scores = None
    show_scores = bool(getattr(cfg, "overlay_show_scores", True))
    show_ids = bool(getattr(cfg, "overlay_show_ids", False))
    forward_stats = _init_forward_stats()
    steps_executed = 0
    region_scores_history = []
    last_region_scores_for_plot = None
    region_scores_shape = None

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                steps_executed += 1
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            if cfg.token_selection_enabled and last_overlay_grid is not None:
                replay_images.append(
                    _apply_overlay(
                        img,
                        last_overlay_grid,
                        last_region_scores if show_scores else None,
                        show_ids=show_ids,
                    )
                )
            else:
                replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_time = time.perf_counter() - start_time
                _update_forward_stats(forward_stats, model, cfg, forward_time)
                action_queue.extend(actions)
                if cfg.token_selection_enabled:
                    overlay_grid, region_scores = _extract_overlay_state(model, cfg)
                    if overlay_grid is not None:
                        last_overlay_grid = overlay_grid
                        last_region_scores = region_scores if show_scores else None
                        replay_images[-1] = _apply_overlay(
                            img,
                            last_overlay_grid,
                            last_region_scores if show_scores else None,
                            show_ids=show_ids,
                        )
                    plot_scores = _extract_region_scores_for_plot(model, cfg)
                    if plot_scores is None:
                        plot_scores = last_region_scores_for_plot
                    if plot_scores is not None:
                        if region_scores_shape is None:
                            region_scores_shape = plot_scores.shape
                        if plot_scores.shape == region_scores_shape:
                            region_scores_history.append(plot_scores)
                            last_region_scores_for_plot = plot_scores
                        else:
                            log_message(
                                f"Skipping region score snapshot (shape changed from {region_scores_shape} "
                                f"to {plot_scores.shape}).",
                                log_file,
                            )

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            steps_executed += 1
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    forward_stats["total_steps"] = steps_executed
    return success, replay_images, forward_stats, region_scores_history


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, forward_stats, region_scores_history = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        mp4_path = save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            rollout_dir=cfg.rollout_dir,
            log_file=log_file,
        )
        if cfg.token_selection_enabled:
            save_region_score_plot(region_scores_history, mp4_path, log_file)

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        for line in _format_forward_stats(forward_stats):
            log_message(line, log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)
    log_message(f"Command: {shlex.join(sys.argv)}", log_file)
    if log_file:
        log_file.write("Config:\n")
        log_file.write(json.dumps(cfg.__dict__, indent=2, sort_keys=True, default=str))
        log_file.write("\n")
        log_file.flush()

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
