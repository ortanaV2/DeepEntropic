import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
import random
import os
import gc
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input layout: particle state + 500 nearest neighbor deltas + global force
INPUT_FEATURE_NAMES = ["x", "y", "vx", "vy"]

for i in range(1, 501):
    INPUT_FEATURE_NAMES.extend([
        f"n{i}_dx", f"n{i}_dy", f"n{i}_dvx", f"n{i}_dvy"
    ])

INPUT_FEATURE_NAMES.extend(["gx", "gy"])

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

def load_sqlite_optimized(db_path, table_name, max_clip=1.0, limit=None):
    """
    RAM-optimized data loading through:
    1. Direct loading into target tensors without intermediate lists
    2. Streaming processing per frame
    3. Immediate garbage collection
    4. Memory-mapped tensors where possible
    """
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Determine number of available frames
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_frames = cursor.fetchone()[0]
    
    if limit is not None and limit < total_frames:
        total_frames = limit

    print(f"Loading {total_frames} frames...")

    # Initial data size estimation for pre-allocation
    cursor.execute(f"SELECT inputs FROM {table_name} LIMIT 1")
    sample_blob = cursor.fetchone()[0]
    sample_size = len(np.frombuffer(sample_blob, dtype=np.float32)) // input_dim
    estimated_total_particles = sample_size * total_frames

    print(f"Estimated particle count: {estimated_total_particles}")

    # Pre-allocate tensors (more efficient than lists)
    try:
        # Attempt direct PyTorch tensor allocation
        x_tensor = torch.empty((estimated_total_particles, input_dim), dtype=torch.float32)
        y_tensor = torch.empty((estimated_total_particles, output_dim), dtype=torch.float32)
        print("Tensors successfully pre-allocated")
    except RuntimeError as e:
        print(f"Pre-allocation failed: {e}")
        # Fallback to progressive allocation
        return load_sqlite_progressive(db_path, table_name, max_clip, limit)

    # Sample random frames when limit is set
    if limit is not None:
        cursor.execute(f"SELECT rowid FROM {table_name} ORDER BY RANDOM() LIMIT {limit}")
        selected_rowids = [row[0] for row in cursor.fetchall()]
    else:
        cursor.execute(f"SELECT rowid FROM {table_name}")
        selected_rowids = [row[0] for row in cursor.fetchall()]

    current_idx = 0
    
    # Streaming processing - one frame at a time
    for frame_num, rowid in enumerate(tqdm(selected_rowids, desc="Loading frames")):
        cursor.execute(f"SELECT inputs, targets FROM {table_name} WHERE rowid = ?", (rowid,))
        row = cursor.fetchone()
        if row is None:
            continue

        inp_blob, tgt_blob = row

        # Direct conversion to NumPy arrays (with copy for write access)
        inputs_raw = np.frombuffer(inp_blob, dtype=np.float32)
        if inputs_raw.size % input_dim != 0:
            continue
        inputs_np = inputs_raw.reshape(-1, input_dim)

        targets_raw = np.frombuffer(tgt_blob, dtype=np.float32)
        if targets_raw.size % inputs_np.shape[0] != 0:
            continue
        full_output_dim = targets_raw.size // inputs_np.shape[0]
        if full_output_dim < output_dim:
            continue
        targets_np = targets_raw.reshape(-1, full_output_dim)[:, :output_dim]

        # Number of particles in this frame
        n_particles = inputs_np.shape[0]
        
        # Resize tensors if necessary
        if current_idx + n_particles > x_tensor.shape[0]:
            new_size = max(x_tensor.shape[0] * 2, current_idx + n_particles)
            x_new = torch.empty((new_size, input_dim), dtype=torch.float32)
            y_new = torch.empty((new_size, output_dim), dtype=torch.float32)
            
            x_new[:current_idx] = x_tensor[:current_idx]
            y_new[:current_idx] = y_tensor[:current_idx]
            
            # Explicitly delete old tensors
            del x_tensor, y_tensor
            gc.collect()
            
            x_tensor, y_tensor = x_new, y_new

        # Clipping and copying in one step (memory-efficient)
        x_tensor[current_idx:current_idx + n_particles] = torch.clamp(
            torch.from_numpy(inputs_np), -max_clip, max_clip
        )
        y_tensor[current_idx:current_idx + n_particles] = torch.clamp(
            torch.from_numpy(targets_np), -max_clip, max_clip
        )
        
        current_idx += n_particles

        # Clear intermediate buffers
        del inputs_np, targets_np, inp_blob, tgt_blob
        
        # Garbage collection every 100 frames
        if frame_num % 100 == 0:
            gc.collect()

    conn.close()

    if current_idx == 0:
        raise ValueError("No data loaded.")

    # Adjust final tensor size
    x_final = x_tensor[:current_idx].contiguous()
    y_final = y_tensor[:current_idx].contiguous()
    
    # Delete temporary tensors
    del x_tensor, y_tensor
    gc.collect()

    print(f"Actually loaded particles: {current_idx}")
    return TensorDataset(x_final, y_final)

def load_sqlite_progressive(db_path, table_name, max_clip=1.0, limit=None):
    """
    Progressive loading for cases where pre-allocation fails
    """
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if limit is not None:
        cursor.execute(f"SELECT rowid FROM {table_name} ORDER BY RANDOM() LIMIT {limit}")
        selected_rowids = [row[0] for row in cursor.fetchall()]
    else:
        cursor.execute(f"SELECT rowid FROM {table_name}")
        selected_rowids = [row[0] for row in cursor.fetchall()]

    # Use smaller chunk sizes
    chunk_size = 10000  # Adjustable based on available RAM
    x_chunks, y_chunks = [], []

    for i in tqdm(range(0, len(selected_rowids), chunk_size), desc="Loading chunks"):
        chunk_rowids = selected_rowids[i:i+chunk_size]
        x_chunk_list, y_chunk_list = [], []
        
        for rowid in chunk_rowids:
            cursor.execute(f"SELECT inputs, targets FROM {table_name} WHERE rowid = ?", (rowid,))
            row = cursor.fetchone()
            if row is None:
                continue

            inp_blob, tgt_blob = row

            inputs_raw = np.frombuffer(inp_blob, dtype=np.float32)
            if inputs_raw.size % input_dim != 0:
                continue
            inputs_np = inputs_raw.reshape(-1, input_dim).copy()  # Copy for write access

            targets_raw = np.frombuffer(tgt_blob, dtype=np.float32)
            if targets_raw.size % inputs_np.shape[0] != 0:
                continue
            full_output_dim = targets_raw.size // inputs_np.shape[0]
            if full_output_dim < output_dim:
                continue
            targets_np = targets_raw.reshape(-1, full_output_dim)[:, :output_dim].copy()  # Copy for write access

            np.clip(inputs_np, -max_clip, max_clip, out=inputs_np)
            np.clip(targets_np, -max_clip, max_clip, out=targets_np)

            x_chunk_list.append(inputs_np)
            y_chunk_list.append(targets_np)

        if x_chunk_list:
            x_chunk = torch.from_numpy(np.concatenate(x_chunk_list, axis=0))
            y_chunk = torch.from_numpy(np.concatenate(y_chunk_list, axis=0))
            x_chunks.append(x_chunk)
            y_chunks.append(y_chunk)

        # Cleanup after each chunk
        del x_chunk_list, y_chunk_list
        gc.collect()

    conn.close()

    if not x_chunks:
        raise ValueError("No data loaded.")

    # Final concatenation
    x_final = torch.cat(x_chunks, dim=0)
    y_final = torch.cat(y_chunks, dim=0)
    
    # Cleanup
    del x_chunks, y_chunks
    gc.collect()

    return TensorDataset(x_final, y_final)

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
    # Enable garbage collection
    gc.enable()
    
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    print("Loading dataset (RAM-optimized)...")
    dataset = load_sqlite_optimized(args.db_path, args.table_name, max_clip=args.max_clip, limit=args.limit)
    data_size = len(dataset)
    print(f"Dataset successfully loaded. Total samples: {data_size}")

    # Force garbage collection after dataset loading
    gc.collect()

    # Train/validation split - in-place to avoid copies
    indices = list(range(data_size))
    random.shuffle(indices)

    split_idx = int(data_size * (1 - args.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # DataLoader with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )

    model = SimpleMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, dropout=args.dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    use_amp = args.use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Starting training with {data_size} samples...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
            print(f"Model state_dict saved with Val Loss {val_loss:.8f}")

            torch.save(model, args.save_model_path)
            print(f"Complete model saved: {args.save_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"No improvement for {epochs_no_improve} epochs. Early stopping.")
            break

        # Optional: Garbage collection after every epoch
        if epoch % 5 == 0:
            gc.collect()

    print("Training finished. Best Val Loss:", best_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAM-optimized MLP training on particle data")

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
    parser.add_argument("--use_amp", type=bool, default=True, help="Mixed Precision (if CUDA available)")
    parser.add_argument("--early_stop_patience", type=int, default=7)
    parser.add_argument("--save_every_n_epochs", type=int, default=0, help="If >0 save checkpoint every N epochs")

    args = parser.parse_args()
    main(args)