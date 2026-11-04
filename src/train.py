"""Entrenamiento de la U-Net con early stopping sobre el Dice de validación."""
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .dataset import augment_batch, iterate_batches, to_tensors
from .losses import bce_dice_loss
from .model import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_probs(model: UNet, x: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Probabilidades (N,H,W) para un tensor de entrada (N,1,H,W)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, x.size(0), batch_size):
        out.append(torch.sigmoid(model(x[s: s + batch_size].to(dev))).squeeze(1).cpu())
    return torch.cat(out).numpy()


def batch_dice(probs: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    pred = probs > thr
    gt = gt.astype(bool)
    inter = (pred & gt).sum(axis=(1, 2))
    denom = pred.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
    return float(((2 * inter + 1e-6) / (denom + 1e-6)).mean())


def train_unet(cfg: Config, Xtr, Ytr, Xva, Yva, verbose: bool = True,
               ckpt_path=None, log_path=None):
    """Entrena y devuelve (modelo con los mejores pesos, historial como DataFrame).

    Los mejores pesos se guardan en ``ckpt_path`` (por defecto
    ``<out_dir>/models/unet_best.pt``). Si se entrega ``log_path``, cada época se
    añade también a ese archivo (útil en ejecuciones desatendidas).
    """
    set_seed(cfg.seed)
    device = get_device()
    xtr, ytr = to_tensors(Xtr, Ytr, device)
    xva, yva = to_tensors(Xva, Yva, device)
    Yva_np = np.asarray(Yva)

    model = UNet(base=cfg.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    torch.backends.cudnn.benchmark = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(cfg.seed)

    ckpt = Path(ckpt_path) if ckpt_path is not None else cfg.models_dir / "unet_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, bad, history = -1.0, 0, []

    def log(msg):
        if verbose:
            print(msg)
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    log(f"Dispositivo: {device} | train={len(xtr)} val={len(xva)} | "
        f"parámetros={sum(p.numel() for p in model.parameters()):,} | épocas={cfg.epochs}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in iterate_batches(xtr, ytr, cfg.batch_size, True, gen):
            if len(xb) < 2:  # BatchNorm necesita >1 muestra
                continue
            xb, yb = augment_batch(xb, yb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = bce_dice_loss(model(xb), yb)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(
                    "La pérdida es NaN/inf. Causas frecuentes: precisión mixta en GPUs GTX 16xx "
                    "(usa Config(amp=False)), tasa de aprendizaje muy alta (lr) o datos con NaN.")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabiliza el entrenamiento
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        val_probs = predict_probs(model, xva)
        val_dice = batch_dice(val_probs, Yva_np, cfg.threshold)
        sched.step(val_dice)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_dice": val_dice, "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - t0})
        log(f"época {epoch:3d} | loss {history[-1]['train_loss']:.4f} | "
            f"Dice val {val_dice:.4f} | {history[-1]['seconds']:.0f}s")

        if val_dice > best:
            best, bad = val_dice, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"Early stopping en la época {epoch} (mejor Dice val = {best:.4f})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, pd.DataFrame(history)
