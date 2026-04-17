import torch

def event_density(ev_step):
    e = ev_step.abs().sum(dim=1, keepdim=True)
    return e / (e.amax(dim=(2,3), keepdim=True) + 1e-6)

def reverse_voxels(ev):
    return torch.flip(ev, dims=[1]) * -1