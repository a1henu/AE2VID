import numpy as np
import torch
import torch.nn.functional as F
import cv2

def flow_to_hsv(flow_xy):
    """flow_xy: [H,W,2], 单位: 像素"""
    u, v = flow_xy[...,0], flow_xy[...,1]
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32), angleInDegrees=True)
    hsv = np.zeros((flow_xy.shape[0], flow_xy.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = (ang / 2).astype(np.uint8)        # [0,360)->[0,180) for OpenCV
    hsv[...,1] = 255
    # 以95分位做归一化，避免少数极大值拉伸
    m95 = np.percentile(mag, 95) + 1e-6
    hsv[...,2] = np.clip(mag / m95 * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_quiver(img_gray, flow_xy, step=16, scale=1.0):
    """在灰度图上画稀疏箭头，返回BGR图"""
    H, W = img_gray.shape
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for y in range(step//2, H, step):
        for x in range(step//2, W, step):
            dx, dy = flow_xy[y, x, 0], flow_xy[y, x, 1]
            tip = (int(x + scale*dx), int(y + scale*dy))
            cv2.arrowedLine(vis, (x,y), tip, (0,255,0), 1, tipLength=0.35)
    return vis

def warp_with_flow(img, flow, mode='forward'):
    """
    img: [1,1,H,W] torch float in [0,1]
    flow: [1,2,H,W] 像素位移
    mode='forward'  解释为 f01: 从 I0 到 I1 的前向光流。合成 I1_hat(x,y) = I0(x - u, y - v)
    mode='backward' 解释为 f10: 从 I1 到 I0 的后向光流。合成 I0_hat(x,y) = I1(x + u, y + v)
    返回: warped, valid_mask
    """
    N, C, H, W = img.shape
    device = img.device
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float(); y = y.float()
    u = flow[:,0]  # [1,H,W]
    v = flow[:,1]
    if mode == 'forward':
        x_s = x[None,...] - u
        y_s = y[None,...] - v
    else:  # 'backward'
        x_s = x[None,...] + u
        y_s = y[None,...] + v

    # 归一化到[-1,1]
    gx = 2.0 * (x_s / (W - 1.0)) - 1.0
    gy = 2.0 * (y_s / (H - 1.0)) - 1.0
    grid = torch.stack((gx, gy), dim=-1)  # [1,H,W,2]
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 有效性mask（采样点在图内）
    valid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    valid = valid.float()
    return warped, valid

def mse_on_mask(a, b, mask):
    # a,b: [1,1,H,W]; mask: [1,H,W] 1有效
    m = (mask>0.5).float()
    denom = m.sum().clamp_min(1.0)
    return ((a-b).abs()*m[None,None,...]).sum()/denom

# f0_path = '/mnt/E/baichenxu/datasets/sta_dyn_dataset/scene_0010/frames/frame_000023.png'
# f1_path = '/mnt/E/baichenxu/datasets/sta_dyn_dataset/scene_0010/frames/frame_000024.png'
# fl_path = '/mnt/E/baichenxu/datasets/sta_dyn_dataset/scene_0010/flow/flow_000024.npy'

f0_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/val/sequence_0000000003/frames/frame_0000000004.png'
f1_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/val/sequence_0000000003/frames/frame_0000000005.png'
fl_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/val/sequence_0000000003/flow/disp01_0000000004.npy'

f0 = cv2.imread(f0_path, cv2.IMREAD_GRAYSCALE)
f1 = cv2.imread(f1_path, cv2.IMREAD_GRAYSCALE)
fl = np.load(fl_path)

f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).float() / 255.0
f1 = torch.from_numpy(f1).unsqueeze(0).unsqueeze(0).float() / 255.0
fl = torch.from_numpy(fl).unsqueeze(0).float()

# === 可视化光流（HSV & 箭头） ===
flow_np = fl.squeeze(0).permute(1,2,0).cpu().numpy()   # [H,W,2]
cv2.imwrite('flow_hsv.png', flow_to_hsv(flow_np))
cv2.imwrite('flow_quiver.png', draw_quiver((f0.squeeze()*255).cpu().numpy().astype(np.uint8), flow_np, step=16, scale=1.0))

# === 用两种方向解释去重采样对齐，并比较误差 ===
with torch.no_grad():
    # 假设 f01: I0->I1
    I1_hat_fwd, valid_fwd = warp_with_flow(f0, fl, mode='forward')
    err_fwd = mse_on_mask(I1_hat_fwd, f1, valid_fwd)

    # 假设 f10: I1->I0
    I0_hat_bwd, valid_bwd = warp_with_flow(f1, fl, mode='backward')
    err_bwd = mse_on_mask(I0_hat_bwd, f0, valid_bwd)

print(f"[检查] 作为前向流 f01 (I0->I1) 的误差: {err_fwd.item():.6f}")
print(f"[检查] 作为后向流 f10 (I1->I0) 的误差: {err_bwd.item():.6f}")
direction = "前向 f01 (I0->I1)" if err_fwd < err_bwd else "后向 f10 (I1->I0)"
print(f"=> 更符合的方向判断：{direction}")

# 导出重采样与误差图，帮你肉眼检查
if err_fwd < err_bwd:
    I1_hat = I1_hat_fwd
    diff = (I1_hat - f1).abs()
    vis_base = f1
    mode_tag = "0to1"
else:
    I1_hat = I0_hat_bwd
    diff = (I1_hat - f0).abs()
    vis_base = f0
    mode_tag = "1to0"

I1_hat_np = (I1_hat.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
vis_base_np = (vis_base.squeeze().cpu().numpy()*255).clip(0,255).astype(np.uint8)
diff_np = (diff.squeeze().cpu().numpy()*255*4).clip(0,255).astype(np.uint8)  # 放大对比

cv2.imwrite(f'warp_{mode_tag}.png', I1_hat_np)
cv2.imwrite(f'warp_target_{mode_tag}.png', vis_base_np)
cv2.imwrite(f'warp_diff_{mode_tag}.png', diff_np)
