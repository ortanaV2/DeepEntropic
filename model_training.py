import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Configuration
INPUT_FEATURE_NAMES = ["x", "y", "vx", "vy", "CF"]
TARGET_FEATURE_NAMES = ["dx", "dy", "vx", "vy"]

def build_feature_indices(feature_names):
    return {name: i for i, name in enumerate(feature_names)}

INPUT_FEATURE_IDX = build_feature_indices(INPUT_FEATURE_NAMES)
TARGET_FEATURE_IDX = build_feature_indices(TARGET_FEATURE_NAMES)

# MLP Model Definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Load SQLite dataset
def load_sqlite_flat(db_path, table_name, max_clip=1.0, limit=None):
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT inputs, targets FROM {table_name}"
    if limit is not None:
        query += f" LIMIT {limit}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    x_list, y_list = [], []

    for inp_blob, tgt_blob in rows:
        inputs = np.frombuffer(inp_blob, dtype=np.float32).reshape(-1, input_dim)
        targets = np.frombuffer(tgt_blob, dtype=np.float32).reshape(-1, output_dim)

        # Clip to stabilize training
        inputs = np.clip(inputs, -max_clip, max_clip)
        targets = np.clip(targets, -max_clip, max_clip)

        x_list.append(inputs)
        y_list.append(targets)

    x_all = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
    y_all = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)

    return TensorDataset(x_all, y_all)

# Evaluation
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    # Load dataset
    dataset = load_sqlite_flat(args.db_path, args.table_name, max_clip=args.max_clip, limit=args.limit)
    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)

    split_idx = int(data_size * (1 - args.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model, Optimizer, Loss
    model = SimpleMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        val_loss = evaluate(model, val_loader, loss_fn)

        print(f"Train Loss: {total_loss / len(train_loader):.8f} | Val Loss: {val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_state_path)
            torch.save(model, args.save_model_path)
            print(f"Model saved at epoch {epoch+1} with val loss {val_loss:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on particle data from SQLite")

    parser.add_argument("--db_path", type=str, default="dataset.db")
    parser.add_argument("--table_name", type=str, default="standard")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_state_path", type=str, default="best_mlp_model.pt")
    parser.add_argument("--save_model_path", type=str, default="full_mlp_model.pt")

    args = parser.parse_args()
    main(args)
