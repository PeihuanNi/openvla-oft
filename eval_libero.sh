#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cuda_visible_devices>"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$1"

MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT="/home/nipeihuan/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
USE_L1_REGRESSION=True
USE_DIFFUSION=False
NUM_DIFFUSION_STEPS_TRAIN=50
NUM_DIFFUSION_STEPS_INFERENCE=50
USE_FILM=False
NUM_IMAGES_IN_INPUT=2
USE_PROPRIO=True
CENTER_CROP=True
NUM_OPEN_LOOP_STEPS=8
LORA_RANK=32
UNNORM_KEY=""
LOAD_IN_8BIT=False
LOAD_IN_4BIT=False

TOKEN_SELECTION_ENABLED=True
TOKEN_PRUNE_ENABLED=True
VISION_PARTIAL_UPDATE_ENABLED=True
REGION_EVAL_INTERVAL=3
GRAD_DENOISE_STEPS=1
GRAD_REGION_MASS=0.7
GRAD_REGION_EMA=0.7
GRAD_KEEP_PREV=False
GRAD_TAU=0.1
GRAD_ALPHA=1.0
GRAD_BETA=1.0
TOKEN_TEMPORAL_THRESHOLD=0.5
TOKEN_SPATIAL_THRESHOLD=0.9
TOKEN_SPATIAL_RADIUS=1
MIN_KEPT_TOKENS=1
MAX_KEPT_TOKENS=128
REGION_PATCH_SIZE=2

TASK_SUITE_NAME="libero_spatial"
NUM_STEPS_WAIT=10
NUM_TRIALS_PER_TASK=50
INITIAL_STATES_PATH="DEFAULT"
ENV_IMG_RES=256

RUN_ID_NOTE=""
LOCAL_LOG_DIR="./experiments/logs"
ROLLOUT_DIR="./rollouts/2026_01_21/speed/"
USE_WANDB=False
WANDB_ENTITY="your-wandb-entity"
WANDB_PROJECT="your-wandb-project"
SEED=7

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
  --region_eval_interval "${REGION_EVAL_INTERVAL}"
  --grad_denoise_steps "${GRAD_DENOISE_STEPS}"
  --grad_region_mass "${GRAD_REGION_MASS}"
  --grad_keep_prev "${GRAD_KEEP_PREV}"
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
