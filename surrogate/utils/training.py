from __future__ import annotations
import math, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_epochs: int = 50,
                min_lr: float = 1e-6,
                save_name: str = "Transformer",
                save_dir: str = "surrogate/ckpt"):
    best_rmse = float("inf")
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for i, (X, F, Y, attn_mask) in enumerate(pbar):
            X, F, Y, attn_mask = X.to(device), F.to(device), Y.to(device), attn_mask.to(device)
            optimizer.zero_grad()
            pred = model(X, F, src_key_padding_mask=attn_mask)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 100 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, F, Y, attn_mask in val_loader:
                X, F, Y, attn_mask = X.to(device), F.to(device), Y.to(device), attn_mask.to(device)
                pred = model(X, F, src_key_padding_mask=attn_mask)
                val_loss += criterion(pred, Y).item()
        val_loss /= len(val_loader)
        val_rmse = math.sqrt(val_loss)

        scheduler.step()
        elapsed = time.time() - start
        lr = scheduler.get_last_lr()[0]

        print(f"[{epoch+1:03d}/{num_epochs}] LR={lr:.6e} | Train={train_loss:.4f} | "
              f"Val={val_loss:.4f} | ValRMSE={val_rmse:.4f} | {elapsed:.1f}s")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), save_dir / f"{save_name}.pth")
            print(f"  >> Best model saved @ {save_dir}/{save_name}.pth  (RMSE={best_rmse:.4f})")

    print(f"Training Finished! Best Val RMSE: {best_rmse:.4f}")


def evaluate_model(model: torch.nn.Module, data_loader: DataLoader) -> tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()

    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    tot_mse = 0.0
    tot_mae = 0.0
    with torch.no_grad():
        for X, F, Y, attn_mask in data_loader:
            X, F, Y, attn_mask = X.to(device), F.to(device), Y.to(device), attn_mask.to(device)
            pred = model(X, F, src_key_padding_mask=attn_mask)
            tot_mse += mse_fn(pred, Y).item()
            tot_mae += mae_fn(pred, Y).item()
    n = len(data_loader)
    avg_mse = tot_mse / n
    avg_mae = tot_mae / n
    rmse = math.sqrt(avg_mse)
    return rmse, avg_mae