import os
import cv2
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.dataset import SeqDataset
from utils.event_utils import event_density, reverse_voxels
from utils.data_utils import SeqCrop128
from utils.loading_utils import get_device
from utils.inference_utils import CropParameters, events_to_voxel_grid_pytorch, events_to_voxel_grid
from model.mixer import Mixer
    
class BiApEVID(nn.Module):
    def __init__(self, adapter, e2vid):
        super().__init__()
        
        self.adapter_fwd = adapter
        self.e2vid_fwd   = e2vid
        self.adapter_bwd = copy.deepcopy(adapter)
        self.e2vid_bwd   = copy.deepcopy(e2vid)
        self.mixer   = Mixer()

        self.device = get_device(use_gpu=True)
        self.height = e2vid.config.get('height', 720)
        self.width  = e2vid.config.get('width', 1280)
        self.num_bins = e2vid.config['num_bins']
        self.crop = CropParameters(self.width, self.height, e2vid.config['num_encoders'])

    def forward(self, f0, f1, evs):
        # f0,f1: [B,1,H,W]; ev: [B,T,Bin,H,W]
        bsz, t, bin, height, width = evs.shape

        # forward process
        f0_vox = f0.repeat(1, bin, 1, 1)
        adapter_i0, st_f = self.adapter_fwd(f0_vox, None)
        fwd = [f0]
        for i in range(t):
            recon, st_f = self.e2vid_fwd(evs[:,i], st_f)
            fwd.append(recon)
        
        # backward process
        f1_vox = f1.repeat(1, bin, 1, 1)
        adapter_i1, st_b = self.adapter_bwd(f1_vox, None)
        bwd = [f1]
        for i in range(t-1, -1, -1):
            rev_ev = reverse_voxels(evs[:,i])
            recon, st_b = self.e2vid_bwd(rev_ev, st_b)
            bwd.append(recon)
        bwd = bwd[::-1]  # reverse to time order
        
        outs, ws = [f0], []
        for i in range(1, t):
            s0 = fwd[i]
            s1 = bwd[i]
            ev = evs[:,i]
            tau = i / t
            tau_tensor = torch.ones((bsz, 1, height, width), device=self.device) * tau
            fused, w = self.mixer(s0, s1, f0, f1, ev, tau_tensor)
            outs.append(fused)
            ws.append(w)
        outs.append(f1)
        
        fused = torch.stack(outs, dim=1)  # [B,T+1,1,H,W]
        fwd = torch.stack(fwd, dim=1)    # [B,T+1,1,H,W]
        bwd = torch.stack(bwd, dim=1)    # [B,T+1,1,H,W]
        return fused, ws, adapter_i0, adapter_i1, fwd, bwd
    
    def initialize_inference(self, f0, f1, evs_list, start_idx, save_dir):
        end_idx = start_idx + len(evs_list)
        forward_dir = os.path.join(save_dir, 'forward')
        backward_dir = os.path.join(save_dir, 'backward')
        os.makedirs(forward_dir, exist_ok=True)
        os.makedirs(backward_dir, exist_ok=True)
        
        fwd_filelist = self.inference_forward(f0, evs_list, start_idx, forward_dir)
        bwd_filelist = self.inference_backward(f1, evs_list, end_idx, backward_dir)
        return fwd_filelist, bwd_filelist
        
    def inference_forward(self, f0, evs_filelist, start_idx, save_dir):
        t = len(evs_filelist)
        self.save_img(os.path.join(save_dir, f'{start_idx:06d}.png'), f0)
        f0_vox = f0.repeat(1, self.num_bins, 1, 1)
        f0_vox = self.crop.pad(f0_vox)
        with torch.no_grad():
            _, st_f = self.adapter_fwd(f0_vox, None)

        fwd_filelist = [os.path.join(save_dir, f'{start_idx:06d}.png')]
        for i in range(t):
            ev = self.load_event_to_voxel(evs_filelist[i], self.num_bins, self.width, self.height, self.device)
            ev = self.crop.pad(ev).unsqueeze(0)
            with torch.no_grad():
                recon, st_f = self.e2vid_fwd(ev, st_f)
                # recon = self.normalize_img(recon)
            self.save_img(os.path.join(save_dir, f'{start_idx + i + 1:06d}.png'), recon[:, :, self.crop.iy0:self.crop.iy1, self.crop.ix0:self.crop.ix1])
            fwd_filelist.append(os.path.join(save_dir, f'{start_idx + i + 1:06d}.png'))
        return fwd_filelist
    
    def inference_backward(self, f1, evs_filelist, end_idx, save_dir):
        t = len(evs_filelist)
        start_idx = end_idx - t
        self.save_img(os.path.join(save_dir, f'{end_idx:06d}.png'), f1)
        f1_vox = f1.repeat(1, self.num_bins, 1, 1)
        f1_vox = self.crop.pad(f1_vox)
        with torch.no_grad():
            _, st_b = self.adapter_bwd(f1_vox, None)
        
        bwd_filelist = [os.path.join(save_dir, f'{end_idx:06d}.png')]
        for i in range(t-1, -1, -1):
            ev = self.load_event_to_voxel(evs_filelist[i], self.num_bins, self.width, self.height, self.device)
            ev = self.crop.pad(ev).unsqueeze(0)
            rev_ev = reverse_voxels(ev)
            with torch.no_grad():
                recon, st_b = self.e2vid_bwd(rev_ev, st_b)
                # recon = self.normalize_img(recon)
            self.save_img(os.path.join(save_dir, f'{start_idx + i:06d}.png'), recon[:, :, self.crop.iy0:self.crop.iy1, self.crop.ix0:self.crop.ix1])
            bwd_filelist.append(os.path.join(save_dir, f'{start_idx + i:06d}.png'))
        bwd_filelist = bwd_filelist[::-1]
        return bwd_filelist
    
    def fuse_frames(self, evs_filelist, forward_filelist, backward_filelist, f0, f1, start_idx, save_dir):
        def _pad_to_hw(x, Hp, Wp, mode='replicate', value=0.0):
            """Pad to (Hp,Wp) on the right/bottom."""
            H, W = x.shape[-2], x.shape[-1]
            pad = (0, Wp - W, 0, Hp - H)  # (left,right,top,bottom)
            if mode == 'constant':
                return F.pad(x, pad, mode=mode, value=value)
            else:
                return F.pad(x, pad, mode=mode)

        last_idx = len(evs_filelist) - 2
        last_img = None

        for i, (evs, fwd, bwd) in enumerate(zip(evs_filelist, forward_filelist, backward_filelist)):
            # ---- load ----
            ev = self.load_event_to_voxel(evs, self.num_bins, self.width, self.height, self.device)
            if ev.dim() == 3:
                ev = ev.unsqueeze(0)  # (1,Bins,H,W)
            s0 = self.load_img(fwd, self.device)  # (1,C,H,W)
            s1 = self.load_img(bwd, self.device)  # (1,C,H,W)

            # 用当前帧的原始尺寸作为裁剪基准
            B, C, H, W = s0.shape
            Hp = (H + 3) // 4 * 4
            Wp = (W + 3) // 4 * 4

            # ---- pad inputs ----
            # images
            s0p = _pad_to_hw(s0, Hp, Wp, mode='replicate')
            s1p = _pad_to_hw(s1, Hp, Wp, mode='replicate')
            f0p = _pad_to_hw(f0, Hp, Wp, mode='replicate')
            f1p = _pad_to_hw(f1, Hp, Wp, mode='replicate')
            # events (zero-pad)
            evp = _pad_to_hw(ev, Hp, Wp, mode='constant', value=0.0)

            # tau：保持原数值常量填充
            tau = i / (len(evs_filelist) - 1)
            tau_tensor = torch.full((1, 1, H, W), float(tau), device=self.device)
            taup = _pad_to_hw(tau_tensor, Hp, Wp, mode='constant', value=float(tau))

            # ---- forward & crop ----
            with torch.no_grad():
                fused_padded, _ = self.mixer(s0p, s1p, f0p, f1p, evp, taup)
            fused = fused_padded[..., :H, :W]   # 裁回原尺寸
            # fused = self.normalize_img(fused)

            # ---- save ----
            self.save_img(os.path.join(save_dir, f'{start_idx + i + 1:06d}.png'), fused, 1.2)

            if i == last_idx:
                last_img = fused

        return last_img

    def inference(self, f0, f1, evs_filelist, start_idx, save_dir):
        fwd_filelist, bwd_filelist = self.initialize_inference(f0, f1, evs_filelist, start_idx, save_dir)

        fuse_dir = os.path.join(save_dir, 'fused')
        os.makedirs(fuse_dir, exist_ok=True)
        last_img = self.fuse_frames(evs_filelist, fwd_filelist[1:], bwd_filelist[1:], f0, f1, start_idx, fuse_dir)
        return last_img

    @staticmethod
    def load_event_to_voxel(ev_path, num_bins, width, height, device):
        if ev_path.endswith('.npz'):
            ev = np.load(ev_path)['arr_0'].astype(np.float32)
        else:
            ev = np.loadtxt(ev_path, dtype=np.float32)
            
        if ev.size == 0 or ev is None:
            ev = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device) 
        else:
            if ev.ndim == 1:
                ev = ev[np.newaxis, :]
            try:
                ev = events_to_voxel_grid_pytorch(ev, num_bins, width, height, device)
            except:
                print(f"Error loading events from {ev_path}.")
            # ev = events_to_voxel_grid(ev, num_bins, width, height)
            # ev = torch.from_numpy(ev).to(device)
        # ev = BiApEVID.normalize_voxel(ev)
        return ev

    @staticmethod
    def load_img(img_path, device):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        return img_tensor

    @staticmethod
    def save_img(save_path, img_tensor, gamma=1.0):
        img_np = img_tensor.squeeze().cpu().detach().numpy()
        img_np = img_np ** gamma
        img_np = (img_np * 255.0).astype(np.uint8)
        cv2.imwrite(save_path, img_np)
        
    @staticmethod
    def normalize_img(img):
        img_min = img.min()
        img_max = img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        return img_norm
    
    @staticmethod
    def normalize_voxel(events):
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            events = mask * (events - mean) / stddev
    