#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cuda_visible_devices>"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$1"  # GPU id(s) passed from CLI, e.g., "0" or "0,1"

MODEL_FAMILY="openvla"  # Model family key (affects preprocessing and action format)
PRETRAINED_CHECKPOINT="/home/nipeihuan/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"  # Checkpoint path
USE_L1_REGRESSION=True  # Use regression action head
USE_DIFFUSION=False  # Use diffusion action expert
NUM_DIFFUSION_STEPS_TRAIN=50  # Diffusion steps used during training
NUM_DIFFUSION_STEPS_INFERENCE=50  # Diffusion steps used during inference
USE_FILM=False  # Enable FiLM conditioning in vision encoder
NUM_IMAGES_IN_INPUT=2  # Number of image views (e.g., full + wrist)
USE_PROPRIO=True  # Include proprioceptive state
CENTER_CROP=True  # Center-crop input images
NUM_OPEN_LOOP_STEPS=8  # Steps per action chunk (open-loop)
LORA_RANK=32  # LoRA rank for adapters (if used)
UNNORM_KEY=""  # Optional dataset key for action un-normalization
LOAD_IN_8BIT=False  # Load model in 8-bit
LOAD_IN_4BIT=False  # Load model in 4-bit

TOKEN_SELECTION_ENABLED=True  # Enable token scoring/selection logic
TOKEN_PRUNE_ENABLED=True  # Enable pruning (if reuse mode is "none")
VISION_PARTIAL_UPDATE_ENABLED=False  # Reuse cached vision tokens on non-eval frames
FLOPS_PROFILE_ENABLED=True  # Enable FLOPs estimate (theoretical, per-forward)
FLOPS_PROFILE_SILENT=True  # No-op for estimate mode
REGION_EVAL_INTERVAL=2  # Evaluate importance every N frames
GRAD_DENOISE_STEPS=1  # Use last N diffusion steps for grad objective
GRAD_REGION_MASS=0.7  # Mass threshold for region selection (normalized)
GRAD_REGION_EMA=0.7  # EMA factor for region score smoothing (empty disables)
GRAD_KEEP_PREV=False  # Union important mask with previous frame
GRAD_SCORE_METHOD="partial_grad"  # full_grad | partial_grad | attn_only
PARTIAL_GRAD_PHI="l2"  # l2 or l1 for partial-grad objective derivative
PARTIAL_GRAD_POS_WEIGHT=1.0  # Weight for position dimensions in partial-grad
PARTIAL_GRAD_GRIP_WEIGHT=1.0  # Weight for gripper dimension in partial-grad
ATTN_SCORE_BETA=2.0  # Exponent for last-layer attention scores (beta > 0)
HEAD_CONSENSUS_GATING_ENABLED=False  # Enable head-consensus gating in partial-grad scoring
HEAD_CONSENSUS_ETA=2.0  # Exponent for head-consensus gating (eta > 0)
HEAD_G_BETA=2.0  # Exponent for head magnitude in partial-grad (beta > 0)
HEAD_G_NORM="sum"  # Head g normalization across heads: sum or max
TOKEN_REUSE_MODE="none"  # none | reuse_kv | reuse_all
GRAD_TAU=0.1  # Gripper change scale for adaptive weighting
GRAD_ALPHA=1.0  # Gripper weight scale (full-grad)
GRAD_BETA=1.0  # Position weight scale (full-grad)
TOKEN_TEMPORAL_THRESHOLD=0.85  # Temporal cosine threshold for background
TOKEN_SPATIAL_THRESHOLD=0.85  # Spatial cosine threshold for background
TOKEN_SPATIAL_RADIUS=1  # Spatial neighbor radius (in tokens)
MIN_KEPT_TOKENS=1  # Per-image minimum kept tokens (region-aware)
MAX_KEPT_TOKENS=128  # Per-image maximum kept tokens (region-aware)
REGION_PATCH_SIZE=1  # Region size in patch tokens (square)

TASK_SUITE_NAME="libero_spatial"  # Task suite name spatial | object | goal | long
NUM_STEPS_WAIT=10  # Initial no-op steps for stabilization
NUM_TRIALS_PER_TASK=5  # Rollouts per task
INITIAL_STATES_PATH="DEFAULT"  # DEFAULT or path to initial states JSON
ENV_IMG_RES=256  # Env render resolution (not policy input size)

RUN_ID_NOTE=""  # Optional note appended to run id
LOCAL_LOG_DIR="./experiments/logs"  # Where to save eval logs
ROLLOUT_DIR="./rollouts/spatial-partial_grad_max_token=128-beta-grip=1-mild-flops/"  # Where to save rollout mp4s
OVERLAY_SHOW_SCORES=False  # Overlay scores on video
OVERLAY_SHOW_IDS=False  # Overlay region id on video
USE_WANDB=False  # Enable W&B logging
WANDB_ENTITY="your-wandb-entity"  # W&B entity
WANDB_PROJECT="your-wandb-project"  # W&B project
SEED=7  # Random seed

args=(
  --model_family "${MODEL_FAMILY}"
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --use_l1_regression "${USE_L1_REGRESSION}"
  --use_diffusion "${USE_DIFFUSION}"
  --num_diffusion_steps_train "${NUM_DIFFUSION_STEPS_TRAIN}"
  --num_diffusion_steps_inference "${NUM_DIFFUSION_STEPS_INFERENCE}"
  --use_film "${USE_FILM}"
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
  --use_proprio "${USE_PROPRIO}"
  --center_crop "${CENTER_CROP}"
  --num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}"
  --lora_rank "${LORA_RANK}"
  --load_in_8bit "${LOAD_IN_8BIT}"
  --load_in_4bit "${LOAD_IN_4BIT}"
  --token_selection_enabled "${TOKEN_SELECTION_ENABLED}"
  --token_prune_enabled "${TOKEN_PRUNE_ENABLED}"
  --vision_partial_update_enabled "${VISION_PARTIAL_UPDATE_ENABLED}"
  --flops_profile_enabled "${FLOPS_PROFILE_ENABLED}"
  --flops_profile_silent "${FLOPS_PROFILE_SILENT}"
  --region_eval_interval "${REGION_EVAL_INTERVAL}"
  --grad_denoise_steps "${GRAD_DENOISE_STEPS}"
  --grad_region_mass "${GRAD_REGION_MASS}"
  --grad_keep_prev "${GRAD_KEEP_PREV}"
  --grad_score_method "${GRAD_SCORE_METHOD}"
  --partial_grad_phi "${PARTIAL_GRAD_PHI}"
  --partial_grad_pos_weight "${PARTIAL_GRAD_POS_WEIGHT}"
  --partial_grad_grip_weight "${PARTIAL_GRAD_GRIP_WEIGHT}"
  --attn_score_beta "${ATTN_SCORE_BETA}"
  --head_consensus_gating_enabled "${HEAD_CONSENSUS_GATING_ENABLED}"
  --head_consensus_eta "${HEAD_CONSENSUS_ETA}"
  --head_g_beta "${HEAD_G_BETA}"
  --head_g_norm "${HEAD_G_NORM}"
  --token_reuse_mode "${TOKEN_REUSE_MODE}"
  --grad_tau "${GRAD_TAU}"
  --grad_alpha "${GRAD_ALPHA}"
  --grad_beta "${GRAD_BETA}"
  --token_temporal_threshold "${TOKEN_TEMPORAL_THRESHOLD}"
  --token_spatial_threshold "${TOKEN_SPATIAL_THRESHOLD}"
  --token_spatial_radius "${TOKEN_SPATIAL_RADIUS}"
  --min_kept_tokens "${MIN_KEPT_TOKENS}"
  --region_patch_size "${REGION_PATCH_SIZE}"
  --task_suite_name "${TASK_SUITE_NAME}"
  --num_steps_wait "${NUM_STEPS_WAIT}"
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}"
  --initial_states_path "${INITIAL_STATES_PATH}"
  --env_img_res "${ENV_IMG_RES}"
  --local_log_dir "${LOCAL_LOG_DIR}"
  --rollout_dir "${ROLLOUT_DIR}"
  --overlay_show_scores "${OVERLAY_SHOW_SCORES}"
  --overlay_show_ids "${OVERLAY_SHOW_IDS}"
  --use_wandb "${USE_WANDB}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --seed "${SEED}"
)

if [[ -n "${UNNORM_KEY}" ]]; then
  args+=(--unnorm_key "${UNNORM_KEY}")
fi
if [[ -n "${RUN_ID_NOTE}" ]]; then
  args+=(--run_id_note "${RUN_ID_NOTE}")
fi
if [[ -n "${GRAD_REGION_EMA}" ]]; then
  args+=(--grad_region_ema "${GRAD_REGION_EMA}")
fi
if [[ -n "${MAX_KEPT_TOKENS}" ]]; then
  args+=(--max_kept_tokens "${MAX_KEPT_TOKENS}")
fi

MUJOCO_GL="osmesa" PYOPENGL_PLATFORM="osmesa" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python experiments/robot/libero/run_libero_eval.py "${args[@]}"
