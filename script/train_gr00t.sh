lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=groot \
  --output_dir=outputs/train/gr00t_custom \
  --job_name=gr00t_custom \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=100000 \
  --policy.push_to_hub=false \
  --wandb.enable=false