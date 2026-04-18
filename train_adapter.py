import argparse
import os
import time
import yaml
import torch
import tqdm
import numpy as np

from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from dataset.dataset import SeqDataset
from model.model import E2VIDRecurrent
from model.hsg import SimpleHSG
from model.loss import TCLoss, LPIPSLoss
from utils.loading_utils import load_model, get_device
from utils.data_utils import SeqCrop128

def _as_float(x, name=""):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            raise ValueError(f"Config field {name} should be numeric, got: {x!r}")
    raise ValueError(f"Config field {name} should be numeric, got type {type(x)}")

def _grad_norm(params):
    norms = []
    for p in params:
        if p is not None and p.grad is not None:
            norms.append(p.grad.detach().norm(2))
    if not norms:
        return 0.0
    stacked = torch.stack(norms)
    return float(stacked.norm(2).item())

def _scalar_value(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

def _append_grad_csv(save_dir, epoch_idx, batch_idx, model_name, named_params):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "grad_log.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("epoch,batch,model,param,grad_abs_mean,grad_abs_max,grad_norm2,weight_norm2\n")
        for name, p in named_params:
            if (p.grad is None) or (not p.requires_grad):
                continue
            g = p.grad.detach()
            g_abs = g.abs()
            g_mean = float(g_abs.mean())
            g_max  = float(g_abs.max())
            g_n2   = float(g.norm(2))
            w_n2   = float(p.detach().norm(2))
            f.write(f"{epoch_idx},{batch_idx},{model_name},{name},{g_mean:.6e},{g_max:.6e},{g_n2:.6e},{w_n2:.6e}\n")

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


def train_one_epoch(
    adapter,
    e2vid,
    train_loader,
    loss_obj,
    loss_weights,
    optimizer,
    device,
    num_bins: int = 5,
    L0: int = 10,
    writer: SummaryWriter = None,
    epoch_idx: int = 0,
    grad_clip: float = 1.0,
    freeze_e2vid: bool = False,
    grad_log_every: int = 0,
):
    adapter.train()
    # e2vid 是否冻结
    if freeze_e2vid:
        e2vid.eval()
        for p in e2vid.parameters():
            p.requires_grad = False
    else:
        e2vid.train()
        for p in e2vid.parameters():
            p.requires_grad = True

    l1loss, lpipsloss, tcloss = loss_obj
    adapter_weight, l1_weight, lpips_weight, tc_weight = loss_weights
    epoch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
    num_batches = len(train_loader)

    pbar = tqdm.tqdm(train_loader, desc=f"[Train] Epoch {epoch_idx+1}", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)   # [B,T+1,1,H,W] 
        events = batch["events"].to(device)   # [B,T,Bin,H,W]
        flows  = batch["flows"].to(device)    # [B,T,2,H,W]

        _, T1, _, _, _ = frames.shape
        T = T1 - 1

        current_L = 0
        prev_states = None
        p0 = None

        batch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
        optimizer.zero_grad(set_to_none=True)

        for t in range(T):
            f0 = frames[:, t]
            f1 = frames[:, t+1]
            ev = events[:, t]
            fl = flows[:, t]

            if t == 0 or prev_states is None:
                f0_vox = f0.repeat(1, num_bins, 1, 1)
                p0, prev_states = adapter(f0_vox, prev_states)
                adapter_loss = l1loss(p0, f0)
                batch_loss["total"] += adapter_weight * adapter_loss
                batch_loss["l1"]    += adapter_weight * adapter_loss

            recon, states = e2vid(ev, prev_states)
            
            l1_loss = l1loss(recon, f1)
            l1_loss = 0
            if current_L < L0:
                tc_loss = 0
                current_L += 1
            else:
                tc_loss = tcloss(f0, f1, p0, recon, fl)
            lpips_loss = lpipsloss(recon, f1, normalize=True)

            # 统计显示
            batch_loss["total"] += (l1_weight * l1_loss + lpips_weight * lpips_loss + tc_weight * tc_loss)
            batch_loss["l1"]    += l1_weight * l1_loss
            batch_loss["lpips"] += lpips_weight * lpips_loss
            batch_loss["tc"]    += tc_weight * tc_loss

            prev_states = states
            p0 = recon 

        batch_loss["total"].backward()
        
        if writer is not None and grad_log_every and (batch_idx % grad_log_every == 0):
           save_dir = getattr(writer, "log_dir", "./logs")
           
           ad_params = [p for p in adapter.parameters() if p.requires_grad]
           ev_params = [p for p in e2vid.parameters() if p.requires_grad]
           gn_adapter = _grad_norm(ad_params)
           gn_e2vid   = _grad_norm(ev_params)
           writer.add_scalar("grads/global_norm_adapter", gn_adapter, epoch_idx * len(train_loader) + batch_idx)
           if not freeze_e2vid and any(p.requires_grad for p in e2vid.parameters()):
               writer.add_scalar("grads/global_norm_e2vid", gn_e2vid, epoch_idx * len(train_loader) + batch_idx)
           # 详细 per-parameter 追加到 CSV（可能较大，按频率控制）
           _append_grad_csv(save_dir, epoch_idx + 1, batch_idx, "adapter", adapter.named_parameters())
           if not freeze_e2vid:
               _append_grad_csv(save_dir, epoch_idx + 1, batch_idx, "e2vid",   e2vid.named_parameters())
               
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + [p for p in e2vid.parameters() if p.requires_grad],
            grad_clip
        )
        optimizer.step()

        # 按时间步取均值，作为“当前 batch 的平均 step 损失”
        batch_avg_total = _scalar_value(batch_loss["total"]) / T
        batch_avg_l1    = _scalar_value(batch_loss["l1"])    / T
        batch_avg_lpips = _scalar_value(batch_loss["lpips"]) / T
        batch_avg_tc    = _scalar_value(batch_loss["tc"])    / T

        # tqdm 显示
        pbar.set_postfix({
            "loss":  f"{batch_avg_total:.4f}",
            "l1":    f"{batch_avg_l1:.4f}",
            "lpips": f"{batch_avg_lpips:.4f}",
            "tc":    f"{batch_avg_tc:.4f}",
        })

        # 累加到 epoch（按 batch 平均）
        epoch_loss["total"] += batch_avg_total
        epoch_loss["l1"]    += batch_avg_l1
        epoch_loss["lpips"] += batch_avg_lpips
        epoch_loss["tc"]    += batch_avg_tc

    # 计算 epoch 平均
    epoch_avg = {k: v / max(1, num_batches) for k, v in epoch_loss.items()}

    # 写 TensorBoard（按 epoch）
    if writer is not None:
        writer.add_scalar("train/epoch_total_loss", epoch_avg["total"], epoch_idx + 1)
        writer.add_scalar("train/epoch_l1_loss",    epoch_avg["l1"],    epoch_idx + 1)
        writer.add_scalar("train/epoch_lpips_loss", epoch_avg["lpips"], epoch_idx + 1)
        writer.add_scalar("train/epoch_tc_loss",    epoch_avg["tc"],    epoch_idx + 1)

        # 记录不同 param group 的 lr
        for gi, g in enumerate(optimizer.param_groups):
            writer.add_scalar(f"train/lr_group{gi}", g["lr"], epoch_idx + 1)

    return epoch_avg  # dict: {'total','lpips','tc'}


@torch.no_grad()
def validate_one_epoch(
    adapter,
    e2vid,
    val_loader,
    loss_obj,
    loss_weights,
    device,
    num_bins: int = 5,
    L0: int = 10,
    writer: SummaryWriter = None,
    epoch_idx: int = 0,
    vis_first_batch: bool = True,
):
    adapter.eval()
    e2vid.eval()
    
    l1loss, lpipsloss, tcloss = loss_obj
    adapter_weight, l1_weight, lpips_weight, tc_weight = loss_weights

    epoch_loss = {"total": 0.0, "l1": 0.0, "lpips": 0.0, "tc": 0.0}
    num_steps_sum = 0

    pbar = tqdm.tqdm(val_loader, desc=f"[Val]   Epoch {epoch_idx+1}", ncols=120)
    first_vis_done = False

    for batch in pbar:
        frames = batch["frames"].to(device)   # [B,T+1,1,H,W]
        events = batch["events"].to(device)   # [B,T,Bin,H,W]
        flows  = batch["flows"].to(device)    # [B,T,2,H,W]

        _, T1, _, _, _ = frames.shape
        T = T1 - 1

        current_L = 0
        prev_states = None
        p0 = None

        # 时间维评估
        for t in range(T):
            f0 = frames[:, t]
            f1 = frames[:, t+1]
            ev = events[:, t]
            fl = flows[:, t]

            if t == 0 or prev_states is None:
                f0_vox = f0.repeat(1, num_bins, 1, 1)
                p0, prev_states = adapter(f0_vox, None)
                adapter_loss = l1loss(p0, f0)
                epoch_loss["total"] += adapter_weight * adapter_loss
                epoch_loss["l1"]    += adapter_weight * adapter_loss

            recon, states = e2vid(ev, prev_states)
            
            # l1_loss = l1loss(recon, f1)
            l1_loss = 0
            if current_L < L0:
                tc_loss = 0
                current_L += 1
            else:
                tc_loss = tcloss(f0, f1, p0, recon, fl)
            lpips_loss = lpipsloss(recon, f1, normalize=True)

            epoch_loss["total"] += float(l1_weight * l1_loss + lpips_weight * lpips_loss + tc_weight * tc_loss)
            epoch_loss["l1"]    += float(l1_weight * l1_loss)
            epoch_loss["lpips"] += float(lpips_weight * lpips_loss)
            epoch_loss["tc"]    += float(tc_weight * tc_loss)
            num_steps_sum += 1

            prev_states = states
            p0 = recon

        # 可选：只在第一个 batch 可视化几张
        if (not first_vis_done) and vis_first_batch and writer is not None:
            try:
                import torchvision.utils as vutils
                grid = vutils.make_grid(
                    torch.cat([f0[:2], recon[:2], f1[:2]], dim=0),  # [6,1,H,W]
                    nrow=2, normalize=True
                )
                writer.add_image("val/vis_f0_recon_f1", grid, epoch_idx + 1)
                first_vis_done = True
            except Exception as e:
                print(f"[val/vis] skip ({e})")

    # 计算 epoch 平均（按 step）
    if num_steps_sum == 0:
        avg = {k: 0.0 for k in epoch_loss}
    else:
        avg = {k: v / num_steps_sum for k, v in epoch_loss.items()}

    # 写 TensorBoard
    if writer is not None:
        writer.add_scalar("val/epoch_total_loss", avg["total"], epoch_idx + 1)
        writer.add_scalar("val/epoch_l1_loss",    avg["l1"],    epoch_idx + 1)
        writer.add_scalar("val/epoch_lpips_loss", avg["lpips"], epoch_idx + 1)
        writer.add_scalar("val/epoch_tc_loss",    avg["tc"],    epoch_idx + 1)

    return avg

def main():
    parser = argparse.ArgumentParser(description="Train Adapter for E2VID Recurrent Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file for the model.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_params   = config["params"]
    train_config   = config["train"]

    save_dir = train_config.get("save_dir", "./logs/adapter_training/")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    device = get_device(use_gpu=True)

    # 模型
    adapter_type = str(model_params.get("adapter_type", "e2vid")).lower()
    if adapter_type == "hsg":
        adapter = SimpleHSG(model_params).to(device)
    else:
        adapter = E2VIDRecurrent(model_params).to(device)
    e2vid = E2VIDRecurrent(model_params).to(device)

    adapter_ckpt_path = model_params.get("adapter_ckpt_path")
    if adapter_ckpt_path and os.path.isfile(adapter_ckpt_path):
        adapter_state = torch.load(adapter_ckpt_path, map_location=device, weights_only=False)
        adapter_state = adapter_state["state_dict"] if "state_dict" in adapter_state else adapter_state
        adapter.load_state_dict(adapter_state, strict=False)
        print(f"[model] Loaded adapter from {adapter_ckpt_path}")
    else:
        print("[model] No adapter ckpt found, training adapter from scratch")
    e2vid_ckpt_path = model_params["e2vid_ckpt_path"]
    e2vid.load_state_dict(torch.load(e2vid_ckpt_path, map_location=device, weights_only=False)["state_dict"])
    print(f"[model] Loaded e2vid from {e2vid_ckpt_path}")
    
    L0 = train_config.get("L0", 10)
    lr_adapter   = _as_float(train_config.get("lr_adapter", 1e-5), "train.lr_adapter")
    lr_e2vid     = _as_float(train_config.get("lr_e2vid",   1e-5), "train.lr_e2vid")
    weight_decay = _as_float(train_config.get("weight_decay", 0.0), "train.weight_decay")

    # 优化器（支持冻结 e2vid）
    params_group = [
        {"params": adapter.parameters(), "lr": lr_adapter},
        {"params": e2vid.parameters(),   "lr": lr_e2vid},
    ]
    if train_config.get("freeze_e2vid", False):
        params_group = [
            {"params": adapter.parameters(), "lr": lr_adapter}
        ]
        print("[opt] e2vid is frozen")

    optimizer = AdamW(params_group, weight_decay=weight_decay)

    # Loss
    loss_obj = (
        L1Loss(),
        LPIPSLoss(use_gpu=True),
        TCLoss(alpha=50.0),
    )
    loss_weights = (
        _as_float(train_config.get("adapter_weight", 1.0), "train.adapter_weight"),
        _as_float(train_config.get("l1_weight", 1.0), "train.l1_weight"),
        _as_float(train_config.get("lpips_weight", 1.0), "train.lpips_weight"),
        _as_float(train_config.get("tc_weight", 5.0), "train.tc_weight"),
    )

    # 数据
    train_dataset = SeqDataset(
        root=train_config["dataroot"], 
        split="train", 
        transform=SeqCrop128(mode="random", pad_if_small=True),
        num_iter=train_config.get("num_iter", 30)
    )
    val_dataset = SeqDataset(
        root=train_config["dataroot"], 
        split="val", 
        transform=SeqCrop128(mode="center", pad_if_small=True),
        num_iter=train_config.get("num_iter", 30)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=True,
    )

    # 训练循环
    epochs     = train_config["num_epochs"]
    save_freq  = train_config.get("save_freq", 10)
    num_bins   = model_params["num_bins"]
    grad_clip  = train_config.get("grad_clip", 1.0)
    freeze_e2  = train_config.get("freeze_e2vid", False)
    
    scheduler = None
    scheduler_config = train_config.get("scheduler", None)
    if scheduler_config is not None:
        if scheduler_config["type"] == "CosineAnnealingLR":
            T_max = int(scheduler_config.get("T_max", epochs))
            eta_min = _as_float(scheduler_config.get("eta_min", 1e-7), "scheduler.eta_min")
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            print(f"[sched] CosineAnnealingLR with T_max={T_max}, eta_min={eta_min}")
        else:
            print(f"[sched] unknown scheduler type: {scheduler_config['type']}, skip.")
            
    best_val = float("inf")
    
    val_avg = {"total": float("inf"), "lpips": float("inf"), "tc": float("inf")}
    for epoch in range(epochs):
        train_avg = train_one_epoch(
            adapter=adapter,
            e2vid=e2vid,
            train_loader=train_loader,
            loss_obj=loss_obj,
            loss_weights=loss_weights,
            optimizer=optimizer,
            device=device,
            num_bins=num_bins,
            L0=L0,
            writer=writer,
            epoch_idx=epoch,
            grad_clip=grad_clip,
            freeze_e2vid=freeze_e2,
            grad_log_every=train_config.get("grad_log_every", 50)
        )

        if (epoch + 1) % train_config.get("val_freq", 10) == 0:
            val_avg = validate_one_epoch(
                adapter=adapter,
                e2vid=e2vid,
                val_loader=val_loader,
                loss_obj=loss_obj,
                loss_weights=loss_weights,
                device=device,
                num_bins=num_bins,
                L0=L0,
                writer=writer,
                epoch_idx=epoch
            )

        print(f"[Epoch {epoch+1}/{epochs}]",
              f"Train(total/l1/lpips/tc): {train_avg['total']:.4f}/{train_avg['l1']:.4f}/{train_avg['lpips']:.4f}/{train_avg['tc']:.4f}")
        
        if scheduler:
            scheduler.step()

        # 保存策略：定频保存 + 追踪最优 val
        if (epoch + 1) % save_freq == 0:
            save_checkpoint(save_dir, epoch + 1, adapter, optimizer)
            if lr_e2vid > 0:
                save_checkpoint(save_dir, epoch + 1, e2vid, optimizer, name=f"e2vid_epoch_{epoch+1:04d}.pth.tar")
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_checkpoint(save_dir, epoch + 1, adapter, optimizer, tag="best")
            if lr_e2vid > 0:
                save_checkpoint(save_dir, epoch + 1, e2vid, optimizer, name="e2vid_best.pth.tar")

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
