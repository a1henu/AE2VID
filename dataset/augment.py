# offline_pack_to_pt.py
from __future__ import annotations
import os, re, math, argparse, hashlib, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

IMG_RE = re.compile(r"^frame_(\d{10})\.png$")
EVT_RE = re.compile(r"^event_tensor_(\d{10})\.npy$")
FLW_RE = re.compile(r"^disp01_(\d{10})\.npy$")

def _seed_from_seq_aug(seq_name: str, aug_id: int) -> int:
    h = hashlib.sha1(seq_name.encode("utf-8")).digest()
    base64 = int.from_bytes(h[:8], "little", signed=False)
    return (base64 ^ (aug_id + 0x9E3779B97F4A7C15)) & 0xFFFFFFFF

def _affine_rotate_grid(h: int, w: int, angle_deg: float, device, dtype) -> torch.Tensor:
    theta = math.radians(angle_deg)
    A = torch.tensor([[ math.cos(theta), -math.sin(theta), 0.0],
                      [ math.sin(theta),  math.cos(theta), 0.0]],
                     dtype=dtype, device=device).unsqueeze(0)  # [1,2,3]
    return F.affine_grid(A, size=(1, 1, h, w), align_corners=False)  # [1,H,W,2]

def _warp(x: torch.Tensor, grid: torch.Tensor, mode="bilinear") -> torch.Tensor:
    return F.grid_sample(x.unsqueeze(0), grid, mode=mode,
                         padding_mode="zeros", align_corners=False).squeeze(0)

def _rotate_flip_frames(frames: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    T1, _, H, W = frames.shape
    grid = _affine_rotate_grid(H, W, angle, frames.device, frames.dtype)
    out = []
    for t in range(T1):
        x = _warp(frames[t], grid, mode="bilinear")
        if flip_h: x = torch.flip(x, dims=[2])
        if flip_v: x = torch.flip(x, dims=[1])
        out.append(x)
    return torch.stack(out, dim=0)

def _rotate_flip_events(events: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    T, _, H, W = events.shape
    grid = _affine_rotate_grid(H, W, angle, events.device, events.dtype)
    out = []
    for t in range(T):
        x = _warp(events[t], grid, mode="bilinear")
        if flip_h: x = torch.flip(x, dims=[2])
        if flip_v: x = torch.flip(x, dims=[1])
        out.append(x)
    return torch.stack(out, dim=0)

def _rotate_flip_flows(flows: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    T, _, H, W = flows.shape
    grid = _affine_rotate_grid(H, W, angle, flows.device, flows.dtype)
    warped = []
    for t in range(T):
        uv = _warp(flows[t], grid, mode="bilinear")  # [2,H,W]
        if flip_h: uv = torch.flip(uv, dims=[2])
        if flip_v: uv = torch.flip(uv, dims=[1])
        warped.append(uv)
    warped = torch.stack(warped, dim=0)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u, v = warped[:, 0], warped[:, 1]
    u_rot = cos_t * u - sin_t * v
    v_rot = sin_t * u + cos_t * v
    if flip_h: u_rot = -u_rot
    if flip_v: v_rot = -v_rot
    return torch.stack([u_rot, v_rot], dim=1)

def _random_crop_same(frames, events, flows, crop, rng: random.Random):
    _, _, H, W = frames.shape
    assert H >= crop and W >= crop, f"size {H}x{W} < crop {crop}"
    top  = 0 if H == crop else rng.randint(0, H - crop)
    left = 0 if W == crop else rng.randint(0, W - crop)
    sl = (..., slice(top, top+crop), slice(left, left+crop))
    return frames[sl], events[sl], flows[sl]

def _center_crop_same(frames, events, flows, crop):
    _, _, H, W = frames.shape
    assert H >= crop and W >= crop
    top, left = (H - crop)//2, (W - crop)//2
    sl = (..., slice(top, top+crop), slice(left, left+crop))
    return frames[sl], events[sl], flows[sl]

def _io_read_seq(seq_path: str, frame_dir: str, event_dir: str, flow_dir: str):
    fdir = os.path.join(seq_path, frame_dir)
    edir = os.path.join(seq_path, event_dir)
    fldir = os.path.join(seq_path, flow_dir)
    frames = sorted([fn for fn in os.listdir(fdir) if IMG_RE.match(fn)])
    evs    = sorted([fn for fn in os.listdir(edir) if EVT_RE.match(fn)])
    flws   = sorted([fn for fn in os.listdir(fldir) if FLW_RE.match(fn)])
    # load
    imgs = []
    for fn in frames:
        im = cv2.imread(os.path.join(fdir, fn), cv2.IMREAD_GRAYSCALE)
        if im is None: raise IOError(f"read fail: {fn}")
        imgs.append(torch.from_numpy(im.astype(np.float32)/255.0).unsqueeze(0))
    frames_t = torch.stack(imgs, dim=0)                         # [T+1,1,H,W]

    evts = [torch.from_numpy(np.load(os.path.join(edir, fn))).float() for fn in evs]
    events_t = torch.stack(evts, dim=0)                         # [T,B,H,W]

    fws = [torch.from_numpy(np.load(os.path.join(fldir, fn))).float() for fn in flws]
    flows_t = torch.stack(fws, dim=0)                           # [T,2,H,W]
    return frames_t, events_t, flows_t

def _io_save_pt(dst_seq_path: str, frames: torch.Tensor, events: torch.Tensor, flows: torch.Tensor):
    os.makedirs(dst_seq_path, exist_ok=True)
    # 三个独立 .pt（更贴近你的需求）
    torch.save(frames.contiguous(), os.path.join(dst_seq_path, "frames.pt"))
    torch.save(events.contiguous(), os.path.join(dst_seq_path, "events.pt"))
    torch.save(flows.contiguous(),  os.path.join(dst_seq_path, "flows.pt"))

def process_split(src_root: str, dst_root: str, split: str,
                  frame_dir: str, event_dir: str, flow_dir: str,
                  n_aug: int, crop_size: int, max_deg: float,
                  p_hflip: float, p_vflip: float,
                  deterministic: bool, center_val: bool):
    src_split = os.path.join(src_root, split)
    dst_split = os.path.join(dst_root, split)
    os.makedirs(dst_split, exist_ok=True)

    seq_names = sorted([d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))])
    for seq_name in tqdm(seq_names, desc=f"[{split}] pack"):
        seq_path = os.path.join(src_split, seq_name)
        frames_t, events_t, flows_t = _io_read_seq(seq_path, frame_dir, event_dir, flow_dir)

        if split == "validation" and center_val:
            f2, e2, fl2 = _center_crop_same(frames_t, events_t, flows_t, crop_size)
            _io_save_pt(os.path.join(dst_split, f"{seq_name}_aug00"), f2, e2, fl2)
            continue

        for aug_id in range(n_aug):
            rng = random.Random(_seed_from_seq_aug(seq_name, aug_id)) if deterministic else random.Random()
            angle  = rng.uniform(-max_deg, max_deg)
            flip_h = rng.random() < p_hflip
            flip_v = rng.random() < p_vflip

            f_aug = _rotate_flip_frames(frames_t, angle, flip_h, flip_v)
            e_aug = _rotate_flip_events(events_t, angle, flip_h, flip_v)
            fl_aug= _rotate_flip_flows (flows_t , angle, flip_h, flip_v)

            f_aug, e_aug, fl_aug = _random_crop_same(f_aug, e_aug, fl_aug, crop_size, rng)

            _io_save_pt(os.path.join(dst_split, f"{seq_name}_aug{aug_id:02d}"),
                        f_aug, e_aug, fl_aug)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--dst_root", required=True)
    ap.add_argument("--frame_dir", default="frames")
    ap.add_argument("--event_dir", default="VoxelGrid-betweenframes-5")
    ap.add_argument("--flow_dir",  default="flow")
    ap.add_argument("--n_aug",     type=int, default=4)
    ap.add_argument("--crop_size", type=int, default=128)
    ap.add_argument("--max_deg",   type=float, default=20.0)
    ap.add_argument("--p_hflip",   type=float, default=0.5)
    ap.add_argument("--p_vflip",   type=float, default=0.5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--center_val",  action="store_true")
    args = ap.parse_args()

    for split in ("train", "validation"):
        if not os.path.isdir(os.path.join(args.src_root, split)):
            print(f"[skip] no split dir: {split}")
            continue
        process_split(
            src_root=args.src_root, dst_root=args.dst_root, split=split,
            frame_dir=args.frame_dir, event_dir=args.event_dir, flow_dir=args.flow_dir,
            n_aug=args.n_aug, crop_size=args.crop_size, max_deg=args.max_deg,
            p_hflip=args.p_hflip, p_vflip=args.p_vflip,
            deterministic=args.deterministic, center_val=args.center_val
        )

if __name__ == "__main__":
    main()
