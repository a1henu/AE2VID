import torch
import cv2
import numpy as np
import torch.nn.functional as F

from model.rife_model.RIFE import Model

def get_rife_model(ckpt_path, device):
    model = Model()
    model.load_model(ckpt_path, -1)
    model.eval()
    model.device()
    return model

def interpolate_frames(model, img0, img1, device):
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img0 = torch.from_numpy(img0.transpose((2, 0, 1))).unsqueeze(0).to(device)
    img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
    # print(img0.shape, img1.shape)
    with torch.no_grad():
        mid_img = model.inference(img0, img1)
    mid_img = mid_img.squeeze(0).cpu().numpy()
    mid_img = mid_img.transpose((1, 2, 0))
    mid_img = cv2.cvtColor(mid_img, cv2.COLOR_BGR2GRAY)
    mid_img = mid_img[:h, :w]
    return mid_img