import cv2
import numpy as np
import torch
import torch.nn.functional as F


def temporal_consistency_loss(image0, image1, processed0, processed1, flow01, alpha=50.0, output_images=False):
    """ Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to estimate a visibility mask.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param processed0: [N x C x H x W] processed image (reconstruction) 0
        :param processed1: [N x C x H x W] processed image (reconstruction) 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
        :param alpha: used for computation of the visibility mask (default: 50.0)
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    #xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx = xx.to(image0.device)
    yy = yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)
    visibility_mask = torch.exp(-alpha * (image1 - image0_warped_to1) ** 2)
    processed0_warped_to1 = F.grid_sample(processed0, warping_grid)

    tc_map = visibility_mask * torch.abs(processed1 - processed0_warped_to1) \
             / (torch.abs(processed1) + torch.abs(processed0_warped_to1) + 1e-5)

    tc_loss = tc_map.mean()

    if output_images:
        additional_output = {'image0': image0,
                             'image1': image1,
                             'image0_warped_to1': image0_warped_to1,
                             'processed0_warped_to1': processed0_warped_to1,
                             'visibility_mask': visibility_mask,
                             'error_map': tc_map}
        return tc_loss, additional_output

    else:
        return tc_loss


def warping_flow_loss(image0, image1, flow01):
    """ Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to derive a flow loss.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)

    tc_map = torch.abs(image1 - image0_warped_to1)

    tc_loss = tc_map.mean()

    return tc_loss


def voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=False):
    """ Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        This function takes an optic flow tensor and uses this to warp the channels of an
        event voxel grid to an image. The variance of this image is the resulting loss.
        :param voxel: [N x C x H x W] input voxel
        :param displacement: [N x 2 x H x W] displacement map from previous flow tensor to current flow tensor
    """
    if reverse_time:
        displacement = -displacement
    v_shape = voxel.size()
    t_width, t_height, t_channels = v_shape[3], v_shape[2], v_shape[1]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(voxel.device).float(), yy.to(voxel.device).float()

    displacement_x = displacement[:, 1, :, :]  # N x H x W
    displacement_y = displacement[:, 0, :, :]  # N x H x W
    displacement_increment = 1.0/(t_channels-1.0)

    voxel_grid_warped = torch.zeros((v_shape[0], 1, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    for i in range(t_channels):
        warp_magnitude_ratio = (1.0-i*displacement_increment) if reverse_time else i*displacement_increment
        #Add displacement to the x coords
        warping_grid_x = xx + displacement_x*warp_magnitude_ratio # N x H x W
        #Add displacement to the y coords
        warping_grid_y = yy + displacement_y*warp_magnitude_ratio # N x H x W
        warping_grid = torch.stack([warping_grid_y, warping_grid_x], dim=3)  # 1 x H x W x 2
        #Normalize the warping grid to between -1 and 1 (necessary for grid_sample API)
        warping_grid[:,:,:,0] = (2.0*warping_grid[:,:,:,0])/(t_height-1)-1.0
        warping_grid[:,:,:,1] = (2.0*warping_grid[:,:,:,1])/(t_width-1)-1.0
        voxel_channel_warped = F.grid_sample(voxel, warping_grid)
        voxel_grid_warped+=voxel_channel_warped[:, i:i+1, :, :]

    variance = voxel_grid_warped.var()
    tc_loss = -variance
    if not reverse_time:
        reverse_tc_loss = voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=True)
        tc_loss += reverse_tc_loss
    if output_images:
        additional_output = {'voxel_grid': voxel,
                             'voxel_grid_warped': voxel_grid_warped}
        return tc_loss, additional_output
    else:
        return tc_loss

def visualize_optical_flow(frame1, blob, width=240, height=180):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    xx, yy = blob[0, ...], blob[1, ...]
    mag, ang = cv2.cartToPolar(xx, yy)

    hsv[..., 0] = (ang * 180 / np.pi) / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    f0_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/train/sequence_0000001058/frames/frame_0000000023.png'
    f1_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/train/sequence_0000001058/frames/frame_0000000024.png'
    fl_path = '/mnt/D/baichenxu/datasets/ApEvid_dataset/train/sequence_0000001058/flow/disp01_0000000023.npy'

    f0 = cv2.imread(f0_path, cv2.IMREAD_GRAYSCALE)
    f1 = cv2.imread(f1_path, cv2.IMREAD_GRAYSCALE)
    fl = np.load(fl_path)
    fl_viz = visualize_optical_flow(f0, fl)
        
    f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).float() / 255.0
    f1 = torch.from_numpy(f1).unsqueeze(0).unsqueeze(0).float() / 255.0
    fl = torch.from_numpy(fl).unsqueeze(0).float()
    
    # tc_loss, additional = temporal_consistency_loss(f0, f1, f0, f1, fl, output_images=True)
    tc_loss = warping_flow_loss(f0, f1, fl)
    # I0, I1, I0_warped, P0_warped, vis_mask, err_map = additional['image0'], additional['image1'], additional['image0_warped_to1'], additional['processed0_warped_to1'], additional['visibility_mask'], additional['error_map']
    print('Temporal consistency loss:', tc_loss.item())
    # cv2.imwrite('outputs/I0.png', (I0.squeeze().cpu().numpy()*255).astype(np.uint8))
    # cv2.imwrite('outputs/I1.png', (I1.squeeze().cpu().numpy()*255).astype(np.uint8))
    # cv2.imwrite('outputs/flow_viz.png', fl_viz)
    # cv2.imwrite('outputs/I0_warped.png', (I0_warped.squeeze().cpu().numpy()*255).astype(np.uint8))
    # cv2.imwrite('outputs/vis_mask.png', (vis_mask.squeeze().cpu().numpy()*255).astype(np.uint8))
    # cv2.imwrite('outputs/err_map.png', (err_map.squeeze().cpu().numpy()*255).astype(np.uint8))
    