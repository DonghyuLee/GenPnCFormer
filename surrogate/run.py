from __future__ import annotations
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from surrogate.data_loader.dataset import (
    scan_and_load, split_datasets, PnCDataset, collate_fn
)
from surrogate.models.pncformer import PnCFormer
from surrogate.utils.training import train_model, evaluate_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--pairs", type=str, default="CA", help="comma-separated folders, e.g., 'TA,CA,SA'")
    ap.add_argument("--sample_size", type=int, default=20000)
    ap.add_argument("--freq_size", type=int, default=500)
    ap.add_argument("--target", type=str, default="dispersion_relation_supercell",
                    choices=["dispersion_relation_unitcell", "dispersion_relation_supercell", "transmittance"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=14, help="pad length (2*cells max)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--save_name", type=str, default="PnCFormer_DR_v1.0")
    return ap.parse_args()

def main():
    args = parse_args()
    selected_folders = [x.strip() for x in args.pairs.split(",") if x.strip()]
    print("Selected folders:", selected_folders)

    # Load data from H5
    X_list, Y_list, F_list = scan_and_load(args.data_dir, selected_folders,
                                           sample_size=args.sample_size,
                                           freq_size=args.freq_size,
                                           target_key=args.target)

    # Split
    (X_tr, Y_tr, F_tr,
     X_va, Y_va, F_va,
     X_te, Y_te, F_te) = split_datasets(X_list, Y_list, F_list,
                                        train_ratio=0.9, valid_ratio=0.05, test_ratio=None)

    # Datasets / Loaders
    ds_tr = PnCDataset(X_tr, F_tr, Y_tr)
    ds_va = PnCDataset(X_va, F_va, Y_va)
    ds_te = PnCDataset(X_te, F_te, Y_te)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, collate_fn=lambda b: collate_fn(b, args.max_seq_len))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, collate_fn=lambda b: collate_fn(b, args.max_seq_len))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, collate_fn=lambda b: collate_fn(b, args.max_seq_len))

    print(f"Train={len(ds_tr)}  Val={len(ds_va)}  Test={len(ds_te)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PnCFormer(
        x_input_dim=6, 
        f_input_dim=1,
        d_model=128, 
        nhead=4,
        num_encoder_layers=4, 
        num_decoder_layers=4,
        dropout=0.0, 
        output_seq_len=args.freq_size
    ).to(device)

    # Optional: torchinfo summary (ignore if unavailable)
    # try:
    #     from torchinfo import summary
    #     summary(model, input_size=((args.batch_size, args.max_seq_len, 6),
    #                                (args.batch_size, args.freq_size)),
    #             device=str(device))
    # except Exception:
    #     pass

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    train_model(model, dl_tr, dl_va, opt, device,
                num_epochs=args.epochs, min_lr=args.min_lr, save_name=args.save_name, save_dir="surrogate/ckpt")

    # Load best and evaluate
    model.load_state_dict(torch.load(f"surrogate/ckpt/{args.save_name}.pth", map_location=device))
    model.to(device)
    rmse, mae = evaluate_model(model, dl_te)
    print(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()