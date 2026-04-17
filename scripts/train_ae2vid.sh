#!/usr/bin/env bash
set -euo pipefail

python train.py adapter --config configs/train_adapter_v2v.yaml
python train.py ae2vid --config configs/train_ae2vid_v2v.yaml
