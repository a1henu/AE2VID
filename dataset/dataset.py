from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from torch.utils.data import Dataset

import numpy as np
import os, re, cv2, torch, math, random, hashlib
import torch.nn.functional as F

_IMG = re.compile(r"^frame_(\d{10})\.png$")
_EVT = re.compile(r"^event_tensor_(\d{10})\.npy$")
_FLW = re.compile(r"^disp01_(\d{10})\.npy$")

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("Could not read image: {}".format(path))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)  # [1,H,W]

def load_flow(path):
    flow = np.load(path)
    return torch.from_numpy(flow).float()     # [2,H,W]

def load_events(path):
    events = np.load(path)
    return torch.from_numpy(events).float()   # [B,H,W]

# ---------- 几何增广工具 ----------

def _affine_rotate_grid(h: int, w: int, angle_deg: float, device, dtype) -> torch.Tensor:
    theta = math.radians(angle_deg)
    A = torch.tensor([[ math.cos(theta), -math.sin(theta), 0.0],
                      [ math.sin(theta),  math.cos(theta), 0.0]],
                     dtype=dtype, device=device).unsqueeze(0)  # [1,2,3]
    return F.affine_grid(A, size=(1, 1, h, w), align_corners=False)  # [1,H,W,2]

def _warp(img: torch.Tensor, grid: torch.Tensor, mode="bilinear") -> torch.Tensor:
    # img: [C,H,W] -> [C,H,W]
    return F.grid_sample(img.unsqueeze(0), grid, mode=mode,
                         padding_mode="zeros", align_corners=False).squeeze(0)

def _rotate_flip_frames(frames: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    # frames: [T+1,1,H,W]
    T1, C, H, W = frames.shape
    grid = _affine_rotate_grid(H, W, angle, frames.device, frames.dtype)
    out = []
    for t in range(T1):
        x = _warp(frames[t], grid, mode="bilinear")
        if flip_h: x = torch.flip(x, dims=[2])
        if flip_v: x = torch.flip(x, dims=[1])
        out.append(x)
    return torch.stack(out, dim=0)

def _rotate_flip_events(events: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    # events: [T,B,H,W] 只做几何
    T, B, H, W = events.shape
    grid = _affine_rotate_grid(H, W, angle, events.device, events.dtype)
    out = []
    for t in range(T):
        x = _warp(events[t], grid, mode="bilinear")
        if flip_h: x = torch.flip(x, dims=[2])
        if flip_v: x = torch.flip(x, dims=[1])
        out.append(x)
    return torch.stack(out, dim=0)

def _rotate_flip_flows(flows: torch.Tensor, angle: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    """
    flows: [T,2,H,W], (u,v)
    先做空间重采样（旋转+位置翻转），再对向量分量做旋转；hflip 令 u 取反，vflip 令 v 取反。
    """
    T, C, H, W = flows.shape
    assert C == 2
    grid = _affine_rotate_grid(H, W, angle, flows.device, flows.dtype)

    warped = []
    for t in range(T):
        uv = _warp(flows[t], grid, mode="bilinear")  # [2,H,W]
        if flip_h: uv = torch.flip(uv, dims=[2])
        if flip_v: uv = torch.flip(uv, dims=[1])
        warped.append(uv)
    warped = torch.stack(warped, dim=0)             # [T,2,H,W]

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u, v = warped[:, 0], warped[:, 1]
    u_rot = cos_t * u - sin_t * v
    v_rot = sin_t * u + cos_t * v
    if flip_h: u_rot = -u_rot
    if flip_v: v_rot = -v_rot
    return torch.stack([u_rot, v_rot], dim=1)       # [T,2,H,W]

@dataclass
class _SeqMeta:
    seq_path: str
    n: int  # number of frames

class SeqDataset(Dataset):
    """
    __getitem__ 返回整条序列：
      frames: [T+1, 1, H, W]
      events: [T,   B, H, W]
      flows : [T,   2, H, W]
    其中 T = n-1

    新增：
      - 几何增广（仅 train 使用）：±max_deg 旋转、水平/垂直翻转
      - 与 SeqCrop128 兼容：先做旋转/翻转，再调用你传入的 transform（裁剪）
      - quadruple=True 时，数据集长度 ×4，(seq_name, aug_id) 决定性采样增强参数
    """
    def __init__(
        self,
        root: str, split: str = "train",
        frame_dir="frames",
        event_dir="voxels",
        flow_dir="flow",
        num_iter: int = 30,
        transform: Optional[Callable] = None,  
        # 增广 & ×4
        use_geom_aug: bool = True,
        aug: int = 4,
        max_deg: float = 20.0,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        quadruple: bool = True,
    ):
        assert split in ("train", "val")
        self.root = os.path.join(root, split)
        self.frame_dir = frame_dir
        self.event_dir = event_dir
        self.flow_dir  = flow_dir
        self.transform = transform

        self.num_iter = num_iter
        self.split = split
        self.use_geom_aug = use_geom_aug
        self.aug = aug
        self.max_deg = max_deg
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.quadruple = quadruple

        self.metas: List[_SeqMeta] = []
        for name in sorted(os.listdir(self.root)):
            seqp = os.path.join(self.root, name)
            if not os.path.isdir(seqp): continue
            fdir, edir, fldir = (os.path.join(seqp, d) for d in (frame_dir, event_dir, flow_dir))
            if not (os.path.isdir(fdir) and os.path.isdir(edir) and os.path.isdir(fldir)):
                continue
            # 可选：也可检查文件完整性
            self.metas.append(_SeqMeta(seqp, self.num_iter))

        if not self.metas:
            raise RuntimeError(f"No valid sequences under {self.root}")

    def __len__(self):
        base = len(self.metas)
        return base * self.aug if (self.quadruple and self.split == "train") else base

    @staticmethod
    def _seed_from_seq_aug(seq_name: str, aug_id: int) -> int:
        """
        用 hashlib.sha1(seq_name) 生成稳定 64bit 种子，并与 aug_id 混合。
        返回值限制在 32bit，兼容 random.Random 的整型种子。
        """
        h = hashlib.sha1(seq_name.encode("utf-8")).digest()  # 20 bytes
        base64 = int.from_bytes(h[:8], "little", signed=False)  # 取前 8 字节
        seed = (base64 ^ (aug_id + 0x9E3779B97F4A7C15)) & 0xFFFFFFFF
        return seed
    
    @staticmethod
    def _normalize_event_voxel(events: torch.Tensor) -> torch.Tensor:
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            events = (events - mean) / stddev
        return events


    def _sample_aug_params(self, seq_name: str, aug_id: int):
        rng = random.Random(self._seed_from_seq_aug(seq_name, aug_id))
        angle  = rng.uniform(-self.max_deg, self.max_deg)
        flip_h = (rng.random() < self.p_hflip)
        flip_v = (rng.random() < self.p_vflip)
        return angle, flip_h, flip_v

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ×4 映射
        aug_id = 0
        if self.quadruple and self.split == "train":
            base_idx = idx // self.aug
            aug_id   = idx % self.aug
        else:
            base_idx = idx

        meta = self.metas[base_idx]
        seqp, n = meta.seq_path, meta.n
        fdir, edir, fldir = (os.path.join(seqp, d) for d in (self.frame_dir, self.event_dir, self.flow_dir))

        frames = torch.stack([load_image(os.path.join(fdir, f"frame_{i:010d}.png")) for i in range(n)], dim=0)     # [T+1,1,H,W]
        events = torch.stack([load_events(os.path.join(edir, f"voxel_{i:010d}.npy")) for i in range(n-1)], dim=0)  # [T,B,H,W]
        flows  = torch.stack([load_flow  (os.path.join(fldir, f"disp01_{i:010d}.npy")) for i in range(n-1)], dim=0)      # [T,2,H,W]

        seq_name = os.path.basename(seqp)

        # 仅训练集做几何增广
        if self.split == "train" and self.use_geom_aug:
            angle, flip_h, flip_v = self._sample_aug_params(seq_name, aug_id)
            frames = _rotate_flip_frames(frames, angle, flip_h, flip_v)
            events = _rotate_flip_events(events, angle, flip_h, flip_v)
            flows  = _rotate_flip_flows (flows , angle, flip_h, flip_v)

        if self.transform is not None:
            frames, events, flows = self.transform(frames, events, flows, seq_name=seq_name)
        events = self._normalize_event_voxel(events)
        
        return {
            "frames": frames,    # [T+1,1,128,128]（若使用 SeqCrop128）
            "events": events,    # [T,B,128,128]
            "flows":  flows,     # [T,2,128,128]
            "T": torch.tensor(n-1, dtype=torch.int32),
            "seq_name": seq_name,
            "aug_id": torch.tensor(aug_id, dtype=torch.int32)
        }



if __name__ == "__main__":
    dataroot = "/mnt/E/baichenxu/datasets/ApEvid_dataset"
    dataset = SeqDataset(root=dataroot, split='train')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Frame0 shape:", sample['frames'].shape)
    print("Event shape:", sample['events'].shape)
    print("Flow shape:", sample['flows'].shape)

    events = sample["events"][0]
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
        print(mean, stddev)
