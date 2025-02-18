#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source $PROJECT_ROOT/.venv/bin/activate

python $SD_SCRIPTS_HOME/sdxl_train_network.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="train_data/" \
  --output_dir="lora/" \
  \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  \
  --resolution=768 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=1200 \
  --dataset_repeats=1 \
  --save_every_n_epochs=1 \
  \
  --mixed_precision="no" \
  --cache_latents \
  --caption_extension=".txt" \
  --enable_bucket \
  --max_data_loader_n_workers=0
