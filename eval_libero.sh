CUDA_VISIBLE_DEVICES="$1" MUJOCO_GL="osmesa" PYOPENGL_PLATFORM="osmesa" python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /home/nipeihuan/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --task_suite_name libero_object \
  --qk_config_json experiments/robot/libero/configs/prune_v2_config.json \
  "${@:2}"
