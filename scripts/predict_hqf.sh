#!/usr/bin/env bash
set -euo pipefail

python predict.py hqf \
  --input_h5 /path/to/HQF_h5/boxes.h5 \
  --delta_frame 112 \
  --recons_ckpt ./pretrained/biape2vid_best.pth.tar \
  --denoiser_ckpt ./pretrained/swinir_idn.pth \
  --rife_ckpt ./pretrained/flownet.pkl \
  --output_dir ./outputs/hqf
