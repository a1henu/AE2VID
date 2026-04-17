#!/usr/bin/env bash
set -euo pipefail

python predict.py real \
  --sequence_dir /path/to/AMED/sequence_0 \
  --width 1280 \
  --height 720 \
  --recons_ckpt ./pretrained/biape2vid_best.pth.tar \
  --output_dir ./outputs/real/sequence_0
