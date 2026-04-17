import argparse
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from model.model import E2VIDRecurrent
from model.BiApEVID import BiApEVID
from model.swinir import SwinIR
from utils.aperture_utils import degrade_img, denoise_img
from utils.inference_utils import CropParameters, events_to_voxel_grid_pytorch, get_device
from utils.interpolate_util import get_rife_model, interpolate_frames


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gamma(x, power):
    return np.clip(x, 0.0, 1.0) ** power


def strip_prefix(state_dict, prefixes=("_orig_mod.", "module.")):
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def get_options(width, height, num_bins=5):
    return {
        "height": height,
        "width": width,
        "num_bins": num_bins,
        "num_encoders": 3,
        "base_num_channels": 32,
        "use_upsample_conv": True,
    }


def load_reconstruction_model(ckpt_path, width, height, num_bins, device):
    options = get_options(width, height, num_bins)
    model = BiApEVID(E2VIDRecurrent(options), E2VIDRecurrent(options)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(strip_prefix(checkpoint["state_dict"]))
    model.eval()
    crop = CropParameters(width, height, options["num_encoders"])
    return model, crop


def load_swinir(ckpt_path, device):
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def save_gray(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))


def tensor_from_gray(image, device):
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)


def denoise_reference(image, denoiser, device, synthesize_fir=True):
    lq = degrade_img(image) if synthesize_fir else image
    return denoise_img(lq, denoiser, device)


def normalize_voxel(voxel):
    nonzero = voxel != 0
    num_nonzero = nonzero.sum()
    if num_nonzero > 0:
        mean = voxel.sum() / num_nonzero
        std = torch.sqrt((voxel ** 2).sum() / num_nonzero - mean ** 2)
        voxel = nonzero.float() * (voxel - mean) / std
    return voxel


def event_slice_h5(events, timestamps, num_bins, width, height, crop, device):
    voxels = []
    for start_ts, end_ts in zip(timestamps[:-1], timestamps[1:]):
        mask = (events[:, 0] >= start_ts) & (events[:, 0] < end_ts)
        voxel = events_to_voxel_grid_pytorch(events[mask], num_bins, width, height, device)
        voxels.append(crop.pad(normalize_voxel(voxel)))
    return torch.stack(voxels, dim=0).unsqueeze(0)


def load_hqf_h5(path):
    with h5py.File(path, "r") as h5_file:
        events = np.transpose(
            np.stack(
                [
                    h5_file["events/ts"],
                    h5_file["events/xs"],
                    h5_file["events/ys"],
                    h5_file["events/ps"],
                ]
            )
        )
        events = np.asarray(sorted(events, key=lambda row: row[0]), dtype=np.float32)

        image_items = []
        for image_name in h5_file["images"]:
            image_items.append((image_name, h5_file["images"][image_name].attrs["timestamp"]))
        image_items = sorted(image_items, key=lambda item: item[1])

        images, frame_ts = [], []
        for image_name, timestamp in image_items:
            images.append(np.asarray(h5_file["images"][image_name]))
            frame_ts.append(timestamp)
    return images, events, frame_ts


@torch.no_grad()
def infer_tensor_window(model, crop, f0, f1, event_voxels, start_idx, end_idx, output_dir, gamma_power):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_gray(output_dir / f"{start_idx:06d}.png", gamma(f0.squeeze().cpu().numpy(), gamma_power))
    save_gray(output_dir / f"{end_idx:06d}.png", gamma(f1.squeeze().cpu().numpy(), gamma_power))

    fused, _, _, _, _, _ = model(crop.pad(f0), crop.pad(f1), event_voxels)
    last_frame = None
    for i in range(1, fused.shape[1] - 1):
        image = fused[:, i, :, crop.iy0:crop.iy1, crop.ix0:crop.ix1].squeeze().cpu().numpy()
        save_gray(output_dir / f"{start_idx + i:06d}.png", gamma(image, gamma_power))
        last_frame = image
    return last_frame, f1.squeeze().cpu().numpy()


def infer_file_window(model, f0, f1, event_files, start_idx, output_dir):
    with torch.no_grad():
        return model.inference(f0, f1, [str(path) for path in event_files], start_idx, str(output_dir))


def predict_evaid(args):
    device = get_device(use_gpu=args.use_gpu)
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_root) / args.sequence
    shape = np.loadtxt(dataset_dir / "shape.txt", dtype=np.int32)
    width, height = int(shape[0]), int(shape[1])
    model, _ = load_reconstruction_model(args.recons_ckpt, width, height, args.num_bins, device)
    denoiser = load_swinir(args.denoiser_ckpt, device)
    rife = get_rife_model(args.rife_ckpt, device) if args.rife_ckpt else None

    event_files = sorted((dataset_dir / "event").iterdir())
    frame_files = sorted((dataset_dir / "gt").iterdir())
    output_dir = Path(args.output_dir) / args.sequence
    start_idx = 0
    while start_idx < len(event_files):
        end_idx = min(start_idx + args.delta_frame, len(event_files) - 1)
        f0 = denoise_reference(read_gray(frame_files[start_idx]), denoiser, device, synthesize_fir=True)
        f1 = denoise_reference(read_gray(frame_files[end_idx]), denoiser, device, synthesize_fir=True)
        save_gray(output_dir / "fused" / f"{start_idx + 1:06d}.png", gamma(f0.squeeze().cpu().numpy(), args.gamma))
        last = infer_file_window(model, f0, f1, event_files[start_idx:end_idx - 1], start_idx + 1, output_dir)
        if rife is not None and last is not None:
            mid = interpolate_frames(rife, last.squeeze().cpu().numpy(), f1.squeeze().cpu().numpy(), device)
            save_gray(output_dir / "fused" / f"{end_idx + 1:06d}.png", gamma(mid, args.gamma))
        start_idx += args.delta_frame


def predict_hqf(args):
    device = get_device(use_gpu=args.use_gpu)
    set_seed(args.seed)

    images, events, frame_ts = load_hqf_h5(args.input_h5)
    height, width = images[0].shape[:2]
    model, crop = load_reconstruction_model(args.recons_ckpt, width, height, args.num_bins, device)
    denoiser = load_swinir(args.denoiser_ckpt, device)
    rife = get_rife_model(args.rife_ckpt, device) if args.rife_ckpt else None
    output_dir = Path(args.output_dir) / Path(args.input_h5).stem

    start_idx = 0
    while start_idx < len(frame_ts):
        end_idx = min(start_idx + args.delta_frame, len(frame_ts) - 1)
        f0 = denoise_reference(images[start_idx].astype(np.float32) / 255.0, denoiser, device, synthesize_fir=True)
        f1 = denoise_reference(images[end_idx].astype(np.float32) / 255.0, denoiser, device, synthesize_fir=True)
        event_voxels = event_slice_h5(events, frame_ts[start_idx:end_idx], args.num_bins, width, height, crop, device)
        last, end_frame = infer_tensor_window(model, crop, f0, f1, event_voxels, start_idx, end_idx, output_dir, args.gamma)
        if rife is not None and last is not None:
            mid = interpolate_frames(rife, last, end_frame, device)
            save_gray(output_dir / f"{end_idx - 1:06d}.png", gamma(mid, args.gamma))
        start_idx += args.delta_frame


def predict_real(args):
    device = get_device(use_gpu=args.use_gpu)
    width, height = args.width, args.height
    model, _ = load_reconstruction_model(args.recons_ckpt, width, height, args.num_bins, device)

    root = Path(args.sequence_dir)
    frame_dir = root / args.frame_dir
    event_dir = root / args.event_dir
    output_dir = Path(args.output_dir)
    f0 = tensor_from_gray(read_gray(frame_dir / args.start_frame), device)
    f1 = tensor_from_gray(read_gray(frame_dir / args.end_frame), device)

    if args.denoiser_ckpt:
        denoiser = load_swinir(args.denoiser_ckpt, device)
        f0 = denoise_img(f0.squeeze().cpu().numpy(), denoiser, device)
        f1 = denoise_img(f1.squeeze().cpu().numpy(), denoiser, device)

    event_files = sorted([p for p in event_dir.iterdir() if p.suffix in {".npz", ".txt"}])
    infer_file_window(model, f0, f1, event_files, args.start_idx, output_dir)


def add_common(parser):
    parser.add_argument("--recons_ckpt", required=True, help="AE2VID reconstruction checkpoint.")
    parser.add_argument("--denoiser_ckpt", default=None, help="SwinIR IDN checkpoint.")
    parser.add_argument("--rife_ckpt", default=None, help="Optional RIFE checkpoint for closing-gap interpolation.")
    parser.add_argument("--output_dir", required=True, help="Directory for predicted frames.")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action="store_true", default=True)


def main():
    parser = argparse.ArgumentParser(description="AE2VID prediction entrypoint")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    evaid = subparsers.add_parser("evaid", help="Predict on an EvAid-style sequence.")
    add_common(evaid)
    evaid.add_argument("--dataset_root", required=True)
    evaid.add_argument("--sequence", required=True)
    evaid.add_argument("--delta_frame", type=int, default=50)

    hqf = subparsers.add_parser("hqf", help="Predict on one HQF h5 file.")
    add_common(hqf)
    hqf.add_argument("--input_h5", required=True)
    hqf.add_argument("--delta_frame", type=int, default=112)

    real = subparsers.add_parser("real", help="Predict on a real AMED-style sequence.")
    add_common(real)
    real.add_argument("--sequence_dir", required=True)
    real.add_argument("--width", type=int, default=1280)
    real.add_argument("--height", type=int, default=720)
    real.add_argument("--frame_dir", default="frames")
    real.add_argument("--event_dir", default="events")
    real.add_argument("--start_frame", default="frame_0.png")
    real.add_argument("--end_frame", default="frame_1.png")
    real.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args()
    if args.dataset in {"evaid", "hqf"} and not args.denoiser_ckpt:
        raise ValueError(f"{args.dataset} prediction requires --denoiser_ckpt for FIR/IDN simulation.")

    if args.dataset == "evaid":
        predict_evaid(args)
    elif args.dataset == "hqf":
        predict_hqf(args)
    elif args.dataset == "real":
        predict_real(args)
    else:
        raise ValueError(f"Unknown dataset mode: {args.dataset}")


if __name__ == "__main__":
    main()
