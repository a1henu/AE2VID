# Checkpoints

Place external checkpoints in this directory or pass their paths through the command line.

Expected files for the released pipeline:

- `biape2vid_best.pth.tar`: AE2VID reconstruction checkpoint.
- `v2v_weight.pth`: V2V-E2VID initialization checkpoint for training.
- `swinir_idn.pth`: SwinIR IDN checkpoint used after FIR.
- `flownet.pkl`: optional RIFE checkpoint for interpolation during the aperture-closing gap.

Large checkpoint files are intentionally ignored by git.
