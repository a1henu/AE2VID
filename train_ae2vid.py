from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from dataset.dataset import SeqDataset
from model.model import E2VIDRecurrent
from model.BiApEVID import BiApEVID
from model.loss import TCLoss, LPIPSLoss
from utils.loading_utils import get_device, load_model
from utils.data_utils import SeqCrop128

import argparse 
import os
import yaml
import torch
import tqdm
import time
import numpy as np

def _as_float(x, name=""):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            raise ValueError(f"Config field {name} should be numeric, got: {x!r}")
    raise ValueError(f"Config field {name} should be numeric, got type {type(x)}")

def _scalar_value(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

def save_checkpoint(save_dir, epoch, model, optimizer, tag=None, name=None):
    os.makedirs(save_dir, exist_ok=True)
    if name is None:
        name = f"adapter_epoch_{epoch:04d}.pth.tar" if tag is None else f"adapter_{tag}.pth.tar"
    path = os.path.join(save_dir, name)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, path)
    print(f"[ckpt] saved → {path}")

def calc_loss(
    adapter_i0, 
    adapter_i1, 
    frames, 
    fused, 
    fwds,
    bwds,
    flows, 
    loss_obj, 
    loss_weights, 
    L0=10,
):
    """
    frames: (B, T+1, 1, H, W)
    recons: (B, T+1, 1, H, W)
    flows:  (B, T, 2, H, W)
    """

    def calc_recons_loss(frames, recons, flows, loss_obj, loss_weights, L0):
        l1_loss, lpips_loss, tc_loss = loss_obj
        adapter_weight, l1_weight, lpips_weight, tc_weight = loss_weights
        
        B, T1, _, H, W = frames.shape
        T = T1 - 1
        
        total_loss = {"l1": 0.0, "lpips": 0.0, "tc": 0.0}
        for i in range(T):
            f0 = frames[:, i]
            f1 = frames[:, i+1]
            r0 = recons[:, i]
            r1 = recons[:, i+1]
            flow = flows[:, i]
            
            total_loss['l1'] += l1_weight * l1_loss(f0, r0)
            total_loss['lpips'] += lpips_weight * lpips_loss(f0, r0, normalize=True)
            if i >= L0:
                total_loss['tc'] += tc_weight * tc_loss(f0, f1, r0, r1, flow)

        total_loss['l1'] += l1_weight * l1_loss(frames[:, -1], recons[:, -1])
        total_loss['lpips'] += lpips_weight * lpips_loss(frames[:, -1], recons[:, -1], normalize=True)
        
        return total_loss
    
    l1_loss, lpips_loss, tc_loss = loss_obj
    adapter_weight, l1_weight, lpips_weight, tc_weight = loss_weights
    
    B, T1, _, H, W = frames.shape
    T = T1 - 1
    
    total_loss = {"adapter": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0, "total": 0.0}
    total_loss['adapter'] += adapter_weight * (l1_loss(adapter_i0, frames[:,0]) + l1_loss(adapter_i1, frames[:,-1]))
    
    fused_loss = calc_recons_loss(frames, fused, flows, loss_obj, loss_weights, L0)
    fwd_loss = calc_recons_loss(frames, fwds, flows, loss_obj, loss_weights, L0)
    bwd_loss = calc_recons_loss(frames, bwds, flows, loss_obj, loss_weights, L0)
    for key in ['l1', 'lpips', 'tc']:
        total_loss[key] += (fused_loss[key] + 0.5 * (fwd_loss[key] + bwd_loss[key]))
        
    total_loss['total'] += total_loss['adapter'] + total_loss['l1'] + total_loss['lpips'] + total_loss['tc']
    
    return total_loss

def train_one_epoch(
    model,
    dataloader,
    loss_obj,
    loss_weights,
    optimizer,
    device,
    epoch_idx,
    writer=None
):
    model.train()
    epoch_loss = {"adapter": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0, "total": 0.0}
    num_batches = len(dataloader)
    pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch_idx+1}")
    for batch_idx, batch in pbar:
        frames = batch['frames'].to(device)  # (B, T+1, 1, H, W)
        events = batch["events"].to(device)  # (B, T, bin, H, W)
        flows  = batch["flows"].to(device)   # (B, T, 2, H, W)
        T = flows.shape[1]
        
        optimizer.zero_grad()
        
        f0, f1 = frames[:,0], frames[:,-1]
        fused, ws, adapter_i0, adapter_i1, fwd, bwd = model(f0, f1, events)
        losses = calc_loss(
            adapter_i0, 
            adapter_i1, 
            frames, 
            fused, 
            fwd, 
            bwd,
            flows, 
            loss_obj, 
            loss_weights
        )
        
        loss = losses['total']
        loss.backward()
        optimizer.step()
        
        batch_avg_total = _scalar_value(losses['total']) / T
        batch_avg_adapter = _scalar_value(losses['adapter']) / 2
        batch_avg_l1    = _scalar_value(losses['l1'])    / T
        batch_avg_lpips = _scalar_value(losses['lpips']) / T
        batch_avg_tc    = _scalar_value(losses['tc'])    / max(1, T - 2)
        pbar.set_postfix({
            "Total Loss": f"{batch_avg_total:.6f}",
            "Adapter Loss": f"{batch_avg_adapter:.6f}",
            "L1 Loss": f"{batch_avg_l1:.6f}",
            "LPIPS Loss": f"{batch_avg_lpips:.6f}",
            "TC Loss": f"{batch_avg_tc:.6f}",
        })
        
        epoch_loss['total']   += batch_avg_total
        epoch_loss['adapter'] += batch_avg_adapter
        epoch_loss['l1']      += batch_avg_l1
        epoch_loss['lpips']   += batch_avg_lpips
        epoch_loss['tc']      += batch_avg_tc
    
    epoch_avg = {k: v / num_batches for k, v in epoch_loss.items()}
    if writer is not None:
        for key, value in epoch_avg.items():
            writer.add_scalar(f"Train/epoch_{key}_loss", value, epoch_idx+1)
        for gi, g in enumerate(optimizer.param_groups):
            writer.add_scalar(f"Train/lr_group{gi}", g["lr"], epoch_idx + 1)
    return epoch_avg

@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    loss_obj,
    loss_weights,
    device,
    epoch_idx,
    writer=None
):
    model.eval()
    epoch_loss = {"adapter": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0, "total": 0.0}
    num_batches = len(dataloader)
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(dataloader), total=num_batches, desc=f"Val Epoch {epoch_idx+1}")
        for batch_idx, batch in pbar:
            frames = batch['frames'].to(device)  # (B, T+1, 1, H, W)
            events = batch["events"].to(device)  # (B, T, bin, H, W)
            flows  = batch["flows"].to(device)   # (B, T, 2, H, W)
            T = flows.shape[1]

            f0, f1 = frames[:,0], frames[:,-1]
            fused, ws, adapter_i0, adapter_i1, fwd, bwd = model(f0, f1, events)
            losses = calc_loss(
                adapter_i0,
                adapter_i1,
                frames,
                fused,
                fwd,
                bwd,
                flows,
                loss_obj,
                loss_weights
            )

            epoch_loss['total']   += _scalar_value(losses['total']) / T
            epoch_loss['adapter'] += _scalar_value(losses['adapter']) / 2
            epoch_loss['l1']      += _scalar_value(losses['l1'])    / T
            epoch_loss['lpips']   += _scalar_value(losses['lpips']) / T
            epoch_loss['tc']      += _scalar_value(losses['tc'])    / max(1, T - 2)

    epoch_avg = {k: v / num_batches for k, v in epoch_loss.items()}
    if writer is not None:
        for key, value in epoch_avg.items():
            writer.add_scalar(f"Val/epoch_{key}_loss", value, epoch_idx+1)
    return epoch_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_params = config["params"]
    model_ckpt = config["model"]
    train_params = config["train"]
    
    save_dir = train_params["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    device = get_device(use_gpu=True)
    
    adapter = E2VIDRecurrent(model_params).to(device)
    e2vid = E2VIDRecurrent(model_params).to(device)

    adapter_ckpt_path = model_ckpt["adapter_ckpt"]
    adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location=device, weights_only=False)["state_dict"])
    e2vid_ckpt_path = model_ckpt["e2vid_ckpt"]
    e2vid.load_state_dict(torch.load(e2vid_ckpt_path, map_location=device, weights_only=False)["state_dict"])
    print(f"[model] Loaded adapter from {adapter_ckpt_path}")
    print(f"[model] Loaded e2vid from {e2vid_ckpt_path}")
    
    BiApEVID_model = BiApEVID(adapter, e2vid).to(device)
    
    lr = _as_float(train_params.get("lr", 1e-4), name="lr")
    weight_decay = _as_float(train_params.get("weight_decay", 0.0), name="weight_decay")
    optimizer = AdamW(BiApEVID_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_obj = (
        L1Loss(),
        LPIPSLoss(),
        TCLoss(alpha=50.0)
    )
    loss_weights = (
        _as_float(train_params.get("adapter_weight", 1.0), name="adapter_weight"),
        _as_float(train_params.get("l1_weight", 1.0), name="l1_weight"),
        _as_float(train_params.get("lpips_weight", 1.0), name="lpips_weight"),
        _as_float(train_params.get("tc_weight", 1.0), name="tc_weight"),
    )
    
    train_dataset = SeqDataset(
        root=train_params["dataroot"], 
        split="train", 
        transform=SeqCrop128(mode="random", pad_if_small=True),
        num_iter=train_params.get("num_iter", 30)
    )
    val_dataset = SeqDataset(
        root=train_params["dataroot"], 
        split="val", 
        transform=SeqCrop128(mode="center", pad_if_small=True),
        num_iter=train_params.get("num_iter", 30)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=train_params.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=train_params.get("num_workers", 4),
        pin_memory=True,
    )
    
    epochs = train_params.get("num_epochs", 100)
    save_freq = train_params.get("save_freq", 5)
    num_bins = model_params.get("num_bins", 5)
    
    scheduler = None
    scheduler_config = train_params.get("scheduler", None)
    if scheduler_config is not None:
        if scheduler_config["type"] == "CosineAnnealingLR":
            T_max = int(scheduler_config.get("T_max", epochs))
            eta_min = _as_float(scheduler_config.get("eta_min", 1e-7), "scheduler.eta_min")
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            print(f"[sched] CosineAnnealingLR with T_max={T_max}, eta_min={eta_min}")
        else:
            print(f"[sched] unknown scheduler type: {scheduler_config['type']}, skip.")
            
    best_val = float("inf")
    val_avg = {"adapter": float("inf"), "l1": float("inf"), "lpips": float("inf"), "tc": float("inf"), "total": float("inf")}
    for epoch in range(epochs):
        train_avg = train_one_epoch(
            BiApEVID_model,
            train_loader,
            loss_obj,
            loss_weights,
            optimizer,
            device,
            epoch,
            writer
        )
        
        if (epoch + 1) % train_params.get("val_freq", 5) == 0:
            val_avg = validate_one_epoch(
                BiApEVID_model,
                val_loader,
                loss_obj,
                loss_weights,
                device,
                epoch,
                writer
            )
        
        print(f"Epoch {epoch+1} Summary:")
        for key, value in train_avg.items():
            print(f"  {key}: {value:.4f}")

        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % save_freq == 0:
            save_checkpoint(save_dir, epoch + 1, BiApEVID_model, optimizer, tag=f"epoch_{epoch+1:04d}", name=f"biape2vid_epoch_{epoch+1:04d}.pth.tar")
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_checkpoint(save_dir, epoch + 1, BiApEVID_model, optimizer, tag="best", name="biape2vid_best.pth.tar")
            
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
