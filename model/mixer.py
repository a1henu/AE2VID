import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, cin, cout, k=3, groups_gn=8):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=k//2, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=max(1, min(groups_gn, cout)), num_channels=cout)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.pw(self.dw(x))))

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        hid = max(4, c // r)
        self.fc1 = nn.Conv2d(c, hid, 1)
        self.fc2 = nn.Conv2d(hid, c, 1)
    def forward(self, x):
        w = x.mean((2,3), keepdim=True)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class EventEncoder(nn.Module):
    """V: (B, Bins, H, W) -> (B, ce+2, H, W)"""
    def __init__(self, bins, ce=32):
        super().__init__()
        self.stem = nn.Sequential(DWConv(bins, ce), DWConv(ce, ce), SE(ce))
        self.stats = nn.Conv2d(bins, 2, 1, bias=False)
        self.down = nn.Conv2d(ce, ce, 3, stride=2, padding=1)
        self.mid  = DWConv(ce, ce)
        self.up   = nn.ConvTranspose2d(ce, ce, 2, stride=2)
    def forward(self, V):
        s = self.stats(V)  
        x = self.stem(V) 
        y = self.up(self.mid(self.down(x)))
        x = x + y
        return torch.cat([x, s], dim=1)

class ImageEncoder(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        b = base
        self.e1 = nn.Sequential(DWConv(in_ch, b), DWConv(b, b), SE(b))
        self.d1 = nn.Conv2d(b, b, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(DWConv(b, 2*b), DWConv(2*b, 2*b), SE(2*b))
        self.d2 = nn.Conv2d(2*b, 2*b, 3, stride=2, padding=1)
        self.e3 = nn.Sequential(DWConv(2*b, 4*b), DWConv(4*b, 4*b), SE(4*b))
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 2, stride=2)
        self.m2 = nn.Sequential(DWConv(4*b, 2*b), SE(2*b))
        self.u1 = nn.ConvTranspose2d(2*b, b, 2, stride=2)
        self.m1 = nn.Sequential(DWConv(2*b, b), SE(b))
        self.out_ch = b
    def forward(self, x):
        e1 = self.e1(x)
        p1 = self.d1(e1)
        e2 = self.e2(p1)
        p2 = self.d2(e2)
        e3 = self.e3(p2)
        u2 = self.u2(e3)
        m2 = self.m2(torch.cat([u2, e2], dim=1))
        u1 = self.u1(m2)
        m1 = self.m1(torch.cat([u1, e1], dim=1))
        return m1

class FusionHead(nn.Module):
    def __init__(self, cin, c_mid, channels=1):
        super().__init__()
        self.fuse = nn.Sequential(DWConv(cin, c_mid), DWConv(c_mid, c_mid), SE(c_mid))
        self.w_head = nn.Conv2d(c_mid, 4, 1) 
        self.r_head = nn.Conv2d(c_mid, channels, 1)  
    def forward(self, feat):
        h = self.fuse(feat)
        w_logits = self.w_head(h)
        r = torch.tanh(self.r_head(h)) * 0.02 
        return w_logits, r

class Mixer(nn.Module):
    def __init__(self, channels=1, event_bins=5, base=32, ce=32):
        super().__init__()
        self.img_enc = ImageEncoder(in_ch=4*channels, base=base)
        self.evt_enc = EventEncoder(bins=event_bins, ce=ce)
        cin = self.img_enc.out_ch + (ce + 2) + 4*channels + 1
        self.head = FusionHead(cin=cin, c_mid=base + ce//2, channels=channels)
        self.channels = channels

    def _as_tau_map(self, tau, H, W, device):
        if isinstance(tau, float) or isinstance(tau, int):
            tau = torch.full((1,1,1,1), float(tau), device=device)
        if tau.dim() == 1:
            tau = tau.view(-1,1,1,1)
        elif tau.dim() == 2:
            tau = tau.view(tau.size(0),1,1,1)
        if tau.size(-1) == 1 or tau.size(-2) == 1:
            tau = tau.expand(-1,1,H,W)
        return tau.to(device)

    def forward(self, Ifwd, Ibwd, B0k, BTk, V, tau):
        B, C, H, W = Ifwd.shape
        device = Ifwd.device

        x_img = torch.cat([Ifwd, Ibwd, B0k, BTk], dim=1)     # (B,4C,H,W)
        fi = self.img_enc(x_img)                             # (B,base,H,W)
        fe = self.evt_enc(V)                                 # (B,ce+2,H,W)

        # 时间通道 (B,1,H,W)
        tau_map = self._as_tau_map(tau, H, W, device).clamp(0.0, 1.0)
        x_raw = x_img                                        # (B,4C,H,W)
        feat = torch.cat([fi, fe, x_raw, tau_map], dim=1)    # (B,cin,H,W)

        w_logits, r = self.head(feat)                        # (B,4,H,W), (B,C,H,W)
        W = torch.softmax(w_logits, dim=1)
        I_stack = torch.stack([Ifwd, Ibwd, B0k, BTk], dim=1) # (B,4,C,H,W)
        I_hat = (W.unsqueeze(2) * I_stack).sum(dim=1) + r
        return I_hat.clamp(0.0, 1.0), W
