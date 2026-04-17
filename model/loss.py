import lpips

from torch import nn
from utils.loss_utils import temporal_consistency_loss

class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg', use_gpu=True):
        super(LPIPSLoss, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn = self.loss_fn.cuda()

    def forward(self, input, target, normalize=False):
        return self.loss_fn(input, target, normalize=normalize).mean()


class TCLoss(nn.Module):
    def __init__(self, alpha=50.0, output_images=False):
        super(TCLoss, self).__init__()
        self.loss = temporal_consistency_loss
        self.alpha = alpha
        self.output_images = output_images

    def forward(self, image0, image1, processed0, processed1, flow01):
        return self.loss(
            image0, image1, 
            processed0, processed1, 
            flow01, alpha=self.alpha, 
            output_images=self.output_images
        )
        

class TotalLoss(nn.Module):
    def __init__(self, l1_weight=1.0, lpips_weight=1.0, tc_weight=5.0, tc_alpha=50.0, use_gpu=True):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = LPIPSLoss(use_gpu=use_gpu)
        self.tc_loss = TCLoss(alpha=tc_alpha)

        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.tc_weight = tc_weight

    def forward(self, image0, image1, processed0, processed1, flow01):
        l1_loss_value = self.l1_loss(processed1, image1)
        lpips_loss_value = self.lpips_loss(processed1, image1)
        tc_loss_value = self.tc_loss(image0, image1, processed0, processed1, flow01)

        total_loss = self.l1_weight * l1_loss_value + self.lpips_weight * lpips_loss_value + self.tc_weight * tc_loss_value
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss_value,
            'lpips_loss': lpips_loss_value,
            'tc_loss': tc_loss_value
        }