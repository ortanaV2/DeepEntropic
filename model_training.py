import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
import random
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input layout: particle state + 3 nearest neighbor deltas + global force
INPUT_FEATURE_NAMES = [
    "x", "y", "vx", "vy",
    "n1_dx", "n1_dy", "n1_dvx", "n1_dvy",
    "n2_dx", "n2_dy", "n2_dvx", "n2_dvy",
    "n3_dx", "n3_dy", "n3_dvx", "n3_dvy",
    "gx", "gy"
]

# Physics simulation outputs: position and velocity changes
TARGET_FEATURE_NAMES = ["dx", "dy", "dvx", "dvy"]

def build_feature_indices(feature_names):
    return {name: i for i, name in enumerate(feature_names)}

INPUT_FEATURE_IDX = build_feature_indices(INPUT_FEATURE_NAMES)
TARGET_FEATURE_IDX = build_feature_indices(TARGET_FEATURE_NAMES)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_sqlite_flat(db_path, table_name, max_clip=1.0, limit=None):
    """
    Loads random frames from SQLite, concatenating all particles into one dataset.
    Each frame contains particle states as 'inputs' and physics updates as 'targets'.
    """
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Sample random frames if limit specified
    cursor.execute(f"SELECT rowid FROM {table_name}")
    all_rowids = [row[0] for row in cursor.fetchall()]
    conn.close()

    if limit is not None and limit < len(all_rowids):
        selected_rowids = random.sample(all_rowids, limit)
    else:
        selected_rowids = all_rowids

    x_list, y_list = [], []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load and reshape particle data from selected frames
    for idx, rowid in enumerate(selected_rowids):
        cursor.execute(f"SELECT inputs, targets FROM {table_name} WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()
        if row is None:
            continue
        inp_blob, tgt_blob = row

        inputs = np.frombuffer(inp_blob, dtype=np.float32)
        if inputs.size % input_dim != 0:
            raise ValueError(f"Inputs blob in row {rowid} not divisible by input_dim ({input_dim}). Got {inputs.size} floats.")
        inputs = inputs.reshape(-1, input_dim)

        full_targets = np.frombuffer(tgt_blob, dtype=np.float32)
        if full_targets.size % inputs.shape[0] != 0:
            raise ValueError(f"Targets blob in row {rowid} incompatible with inputs rows: targets size {full_targets.size}, inputs rows {inputs.shape[0]}.")
        full_output_dim = full_targets.size // inputs.shape[0]
        full_targets = full_targets.reshape(-1, full_output_dim)

        if full_output_dim < output_dim:
            raise ValueError(f"Row {rowid} targets have fewer dims ({full_output_dim}) than expected ({output_dim}).")
        targets = full_targets[:, :output_dim]

        # Clip extreme values for training stability
        inputs = np.clip(inputs, -max_clip, max_clip)
        targets = np.clip(targets, -max_clip, max_clip)

        x_list.append(inputs)
        y_list.append(targets)

    conn.close()

    if len(x_list) == 0:
        raise ValueError("No data loaded from database.")

    x_all = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
    y_all = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)

    return TensorDataset(x_all, y_all)

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)

def main(args):
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    print("Loading dataset...")
    dataset = load_sqlite_flat(args.db_path, args.table_name, max_clip=args.max_clip, limit=args.limit)
    data_size = len(dataset)
    print(f"Total samples (particles across frames): {data_size}")

    # Train/validation split
    indices = list(range(data_size))
    random.shuffle(indices)

    split_idx = int(data_size * (1 - args.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = SimpleMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, dropout=args.dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = loss_fn(pred, y)

            # Skip corrupted batches
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf loss detected, skipping batch")
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(train_loss=total_loss / max(1, n_batches))

        val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / max(1, n_batches):.8f} | Val Loss: {val_loss:.8f}")

        scheduler.step(val_loss)

        # Save best model based on validation loss
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            epochs_no_improve = 0

            os.makedirs(os.path.dirname(args.save_state_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_state_path)
            print(f"Model state_dict saved with val loss {val_loss:.8f}")

            torch.save(model, args.save_model_path)
            print(f"Full model saved to {args.save_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"No improvement for {epochs_no_improve} epochs. Early stopping.")
            break

    print("Training finished. Best val loss:", best_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on particle data from SQLite")

    parser.add_argument("--db_path", type=str, default="dataset.db")
    parser.add_argument("--table_name", type=str, default="standard")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_state_path", type=str, default="best_mlp_model.pt")
    parser.add_argument("--save_model_path", type=str, default="full_mlp_model.pt")

    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--use_amp", type=bool, default=True, help="Use mixed precision (if CUDA available)")
    parser.add_argument("--early_stop_patience", type=int, default=7)
    parser.add_argument("--save_every_n_epochs", type=int, default=0, help="If >0 save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)