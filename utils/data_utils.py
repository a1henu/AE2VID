import torch, random, hashlib
import torch.nn.functional as F

class SeqCrop128:
    """
    对一条序列做对齐裁剪到 128x128：
      - frames: [T+1, C1, H, W]
      - events: [T,   B,  H, W]
      - flows : [T,   2,  H, W]
    仅做裁剪，不缩放，因此光流数值 (u,v) 不变。
    """
    def __init__(self,
                 mode: str = "random",          # "random" | "center"
                 deterministic_by_seq: bool = True,
                 pad_if_small: bool = False,    # 如果 H 或 W < 128，则先 pad 到 >=128 再裁
                 pad_value_frames: float = 0.0,
                 pad_value_events: float = 0.0,
                 pad_value_flows:  float = 0.0):
        assert mode in ("random", "center")
        self.mode = mode
        self.det_by_seq = deterministic_by_seq
        self.pad_if_small = pad_if_small
        self.pad_value_frames = pad_value_frames
        self.pad_value_events = pad_value_events
        self.pad_value_flows  = pad_value_flows

    @staticmethod
    def _seed_from_seq(seq_name: str) -> int:
        h = hashlib.sha1(seq_name.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little")

    @staticmethod
    def _pad_to_minHW(x: torch.Tensor, minH: int, minW: int, value: float) -> torch.Tensor:
        # x: [..., H, W]  -> pad 到至少 minH x minW，右下方向补零（或给定值）
        H, W = x.shape[-2:]
        pad_h = max(0, minH - H)
        pad_w = max(0, minW - W)
        if pad_h == 0 and pad_w == 0:
            return x
        # F.pad 的顺序是 (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), value=value)

    def __call__(self,
                 frames: torch.Tensor,   # [T+1,1,H,W] 或 [T+1,C,H,W]
                 events: torch.Tensor,   # [T,B,H,W]
                 flows:  torch.Tensor,   # [T,2,H,W]
                 seq_name: str = None):
        assert frames.dim() == 4 and events.dim() == 4 and flows.dim() == 4, "输入维度不符"
        H, W = frames.shape[-2], frames.shape[-1]
        assert events.shape[-2:] == (H, W) and flows.shape[-2:] == (H, W), "三者空间尺寸不一致"

        # 需要的话先 pad 到 >= 128
        if self.pad_if_small and (H < 128 or W < 128):
            frames = self._pad_to_minHW(frames, 128, 128, self.pad_value_frames)
            events = self._pad_to_minHW(events, 128, 128, self.pad_value_events)
            flows  = self._pad_to_minHW(flows,  128, 128, self.pad_value_flows)
            H, W = frames.shape[-2], frames.shape[-1]

        # 现在必须能裁出 128 x 128
        if H < 128 or W < 128:
            raise ValueError(f"Sequence spatial size {H}x{W} < 128x128; "
                             f"设 pad_if_small=True 或先做 resize。")

        # 确定裁剪窗口
        if self.mode == "center":
            top  = (H - 128) // 2
            left = (W - 128) // 2
        else:
            if self.det_by_seq and (seq_name is not None):
                rng = random.Random(self._seed_from_seq(seq_name))
                top  = 0 if H == 128 else rng.randint(0, H - 128)
                left = 0 if W == 128 else rng.randint(0, W - 128)
            else:
                top  = 0 if H == 128 else random.randint(0, H - 128)
                left = 0 if W == 128 else random.randint(0, W - 128)

        sl = (..., slice(top, top + 128), slice(left, left + 128))
        frames = frames[sl]
        events = events[sl]
        flows  = flows[sl]
        # 裁剪不会改变 flow 的 (u,v) 数值，仅空间截取

        return frames, events, flows


