import torch
import cv2
import numpy
import os
from torch.utils.data import Dataset
from utils.inference_utils import CropParameters, EventPreprocessor
from torch.nn.functional import grid_sample



class EventSequences(Dataset):

    def __init__(self, root):
        self.sequences = []
        for seq in sorted(os.listdir(root)):
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class EventData:

    def __init__(self, root, seq, width, height, num_encoders, options, transform=None):

        # This members are used to preprocess input events: padding and normalization
        self.crop = CropParameters(width, height, num_encoders)
        self.event_preprocessor = EventPreprocessor(options)

        self.root = root
        self.event_dir = root+seq+'/VoxelGrid-betweenframes-5'
        self.event_tensors = []
        for f in sorted(os.listdir(self.event_dir)):
            if f.endswith('npy'):
                self.event_tensors.append(f)
        if len(self.event_tensors) % 2 != 0:
            self.event_tensors.pop()

        self.frame_dir = root+seq+'/frames'
        self.frames = []
        for f in sorted(os.listdir(self.frame_dir)):
            if f.endswith('png'):
                self.frames.append(f)
        self.transform = transform

        self.flow_dir = root+seq+'/flow'
        self.flow = []
        for f in sorted(os.listdir(self.flow_dir)):
            if f.endswith('npy'):
                self.flow.append(f)

    def get_item(self, index, num_encoders):
        #Take next event/image (index+1) in order to use warp to index -> index+1 to obtain warped image at index+1
        #The flow of index->index+1 is at flow[index]
        event_name = os.path.join(self.event_dir, self.event_tensors[index+1])
        frame_name = os.path.join(self.frame_dir, self.frames[index+1])
        flow_name = os.path.join(self.flow_dir, self.flow[index])

        event_array = numpy.load(event_name)
        event_tensor = torch.tensor(event_array)
        events = event_tensor.unsqueeze(dim=0)
        events = self.event_preprocessor(events)
        events = self.crop.pad(events)

        frame = cv2.imread(frame_name)
        frame_tensor = torch.tensor(numpy.transpose(frame, (2, 0, 1)))
        frame_tensor = frame_tensor.type(torch.FloatTensor)
        frame_t = frame_tensor.unsqueeze(dim=0)
        frame_t = self.crop.pad(frame_t)

        flow_array = numpy.load(flow_name)
        flow_tensor = torch.tensor(flow_array)
        flow_tensor = flow_tensor.type(torch.FloatTensor)
        flow = flow_tensor.unsqueeze(dim=0)
        flow = self.crop.pad(flow)
        flow = flow.permute(0, 2, 3, 1)

        return events, frame_t, flow

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='border'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """

    B, C, H, W = x.size()
    assert x.size()[-2:] == flow.size()[1:3]

    # Mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x) # If x is a tensor in 'cuda:0', grid is too
    vgrid = grid + flow

    # Scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)

    return output
