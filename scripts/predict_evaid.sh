#!/usr/bin/env bash
set -euo pipefail

python predict.py evaid \
  --dataset_root /path/to/EvAid \
  --sequence bear \
  --delta_frame 50 \
  --recons_ckpt ./pretrained/biape2vid_best.pth.tar \
  --denoiser_ckpt ./pretrained/swinir_idn.pth \
  --rife_ckpt ./pretrained/flownet.pkl \
  --output_dir ./outputs/evaid
