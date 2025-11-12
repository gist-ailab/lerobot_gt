lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolVLA \
  --job_name=smol_VLA \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=100000 \
  --policy.push_to_hub=false \
  --wandb.enable=true