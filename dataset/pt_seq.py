# pt_dataset.py
from __future__ import annotations
import os
from typing import Dict, Optional, Callable, List
import torch
from torch.utils.data import Dataset

class PTSeqDataset(Dataset):
    """
    读取每个 seq 目录下的 3 个 .pt 文件：
      frames.pt: [T+1,1,H,W] float32 (0..1)
      events.pt: [T,B,H,W]   float32
      flows.pt : [T,2,H,W]   float32
    """
    def __init__(self,
                 root: str,               # 包含 train / validation
                 split: str = "train",
                 num_iter: int = 30,
                 transform: Optional[Callable] = None):
        assert split in ("train", "validation")
        self.root = os.path.join(root, split)
        self.transform = transform
        self.num_iter = num_iter

        # 一个样本 = 一个子目录（必须含 3 个 pt 文件）
        all_dirs = sorted([d for d in os.listdir(self.root)
                           if os.path.isdir(os.path.join(self.root, d))])
        self.seq_dirs: List[str] = []
        for d in all_dirs:
            p = os.path.join(self.root, d)
            if (os.path.isfile(os.path.join(p, "frames.pt")) and
                os.path.isfile(os.path.join(p, "events.pt")) and
                os.path.isfile(os.path.join(p, "flows.pt"))):
                self.seq_dirs.append(d)

        if not self.seq_dirs:
            raise RuntimeError(f"No packed sequences found under {self.root}")

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.seq_dirs[idx]
        p = os.path.join(self.root, name)
        frames = torch.load(os.path.join(p, "frames.pt"), map_location="cpu")  # [T+1,1,H,W]
        events = torch.load(os.path.join(p, "events.pt"), map_location="cpu")  # [T,B,H,W]
        flows  = torch.load(os.path.join(p, "flows.pt"),  map_location="cpu")  # [T,2,H,W]

        if self.transform is not None:
            frames, events, flows = self.transform(frames, events, flows, seq_name=name)

        return {
            "frames": frames[:self.num_iter + 1, :, :, :],    # [T+1,1,H,W]（通常已是 128x128）
            "events": events[:self.num_iter, :, :, :],    # [T,B,H,W]
            "flows":  flows[:self.num_iter, :, :, :],     # [T,2,H,W]
            "T": torch.tensor(self.num_iter, dtype=torch.int32),
            "seq_name": name,
        }


if __name__ == "__main__":
    # 简单测试
    dataset = PTSeqDataset(root="/mnt/nas-cp/baichenxu/EvHDR/data/ecoco_depthmaps_test_pt", split="validation")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample.items()})