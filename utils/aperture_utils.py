import numpy as np
import h5py
import torch
import cv2

from degradation.degradation import Degradation

def degrade_img(HQ_img):
    parameters = { 
        'gamma': 2.2, 
        'C_min': 0.1, 'C_max': 0.4, 
        'C_var_range': (0.02, 0.08), 
        'time_err_range': (10, 1000), 
        'hot_pixels_prob': 0, 
        'cold_pixels_prob': 0, 
        'gaussian_noise_range': (1/255, 25/255), 
        'salt_pepper_prob': 0, 
        'alpha': 1.0, 
        't_end': 3e5, 
        'enable_hot_cold': True,
        'enable_down': False,
        'down_factors': [(0.5, 2)],
        
        'rng_seed': None
    }
    degradation = Degradation(**parameters)
    LQ_img = degradation.degrade(HQ_img, apply_gamma_correction=False)
    return LQ_img

def denoise_img(LQ_img, denoiser, device):
    LQ_tensor = torch.from_numpy(LQ_img).unsqueeze(0).unsqueeze(0).float().to(device)
    LQ_tensor = LQ_tensor.repeat(1,3,1,1)
    with torch.no_grad():
        denoised_img = denoiser(LQ_tensor)
    # print(denoised_img.min().item(), denoised_img.max().item(), denoised_img.mean().item())
    denoised_img = torch.nn.functional.interpolate(denoised_img, scale_factor=0.25, mode='bicubic', align_corners=False, antialias=True)
    denoised_img = 0.299 * denoised_img[:,0:1,:,:] + 0.587 * denoised_img[:,1:2,:,:] + 0.114 * denoised_img[:,2:3,:,:]
    return denoised_img.clamp(0.0, 1.0)
    # denoise_img = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min())
    # return denoise_img