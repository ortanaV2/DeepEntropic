import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Configuration
INPUT_FEATURE_NAMES = ["x", "y", "vx", "vy"]
TARGET_FEATURE_NAMES = ["dx", "dy"]

# Map feature names to column indices
def build_feature_indices(feature_names):
    return {name: i for i, name in enumerate(feature_names)}

INPUT_FEATURE_IDX = build_feature_indices(INPUT_FEATURE_NAMES)
TARGET_FEATURE_IDX = build_feature_indices(TARGET_FEATURE_NAMES)

# Graph Neural Network Model Definition
class SimpleGNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(aggr='mean')
        # Node feature embedding network
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Edge message function operating on concatenated node embeddings
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Output MLP predicts position displacements
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        # Initial node embedding
        x = self.node_mlp(x)
        # Message passing with aggregation
        x = self.propagate(edge_index, x=x)
        # Predict output deltas
        return self.out_mlp(x)

    def message(self, x_i, x_j):
        # Compose edge messages from source and target node embeddings
        edge_input = torch.cat([x_i, x_j], dim=1)
        return self.edge_mlp(edge_input)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Load dataset from SQLite, convert blobs to PyG Data objects
def load_sqlite_data(db_path, table_name, radius, max_clip=1.0, limit=None):
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

    dataset = []
    for inp_blob, tgt_blob in rows:
        inputs = np.frombuffer(inp_blob, dtype=np.float32).reshape(-1, input_dim)
        targets = np.frombuffer(tgt_blob, dtype=np.float32).reshape(-1, output_dim)

        # Clip inputs and targets to stabilize training
        inputs = np.clip(inputs, -max_clip, max_clip)
        targets = np.clip(targets, -max_clip, max_clip)

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        # Use first two features as position for graph construction
        pos = torch.tensor(inputs[:, :2], dtype=torch.float32)
        edge_index = radius_graph(pos, r=radius, loop=False)

        dataset.append(Data(x=x, edge_index=edge_index, y=y))

    return dataset

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    input_dim = len(INPUT_FEATURE_NAMES)
    output_dim = len(TARGET_FEATURE_NAMES)

    # Load dataset from SQLite
    dataset = load_sqlite_data(
        args.db_path,
        args.table_name,
        radius=args.radius,
        max_clip=args.max_clip,
        limit=args.limit
    )
    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - args.val_split))
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model and optimizer
    model = SimpleGNN(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)

            # Skip batches with invalid loss values
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
    parser = argparse.ArgumentParser(description="Train GNN on particle position data from SQLite")

    parser.add_argument("--db_path", type=str, default="dataset.db",
                        help="SQLite database path")
    parser.add_argument("--table_name", type=str, default="cosmic_2000p_6f_orion",
                        help="SQLite table name containing inputs and targets blobs")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden layer size for MLPs")
    parser.add_argument("--radius", type=float, default=0.1,
                        help="Radius for graph edge construction")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm threshold")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--max_clip", type=float, default=1.0,
                        help="Clipping threshold for inputs and targets")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows loaded from database")
    parser.add_argument("--save_state_path", type=str, default="best_gnn_model.pt",
                        help="Path to save model state dict")
    parser.add_argument("--save_model_path", type=str, default="full_gnn_model.pt",
                        help="Path to save full model")

    args = parser.parse_args()
    main(args)
