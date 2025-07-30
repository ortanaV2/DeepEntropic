import argparse
import sqlite3
import torch
import torch.nn as nn
import datetime
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FullRAMParticleDataset(Dataset):
    def __init__(self, db_path, max_frames=None):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT COUNT(*) FROM drop_10p_dataset")
        total_samples = self.cursor.fetchone()[0]

        if max_frames is not None and max_frames < total_samples:
            self.total_samples = max_frames
        else:
            self.total_samples = total_samples

        self.features_per_particle = 4

        query = f"SELECT inputs, targets FROM drop_10p_dataset LIMIT {self.total_samples}"
        rows = self.cursor.execute(query).fetchall()

        self.data = []
        for inputs_blob, targets_blob in rows:
            inputs = torch.frombuffer(inputs_blob, dtype=torch.float32)
            targets = torch.frombuffer(targets_blob, dtype=torch.float32)
            self.data.append((inputs, targets))

        self.num_particles = len(self.data[0][0]) // self.features_per_particle if self.data else 0

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.float(), y.float()

class ParticleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def train(db_path, max_frames=None, hidden_size=512, epochs=50, batch_size=64, lr=1e-4, early_stop_patience=5):
    dataset = FullRAMParticleDataset(db_path, max_frames=max_frames)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    FEATURES_IN = dataset.features_per_particle
    FEATURES_OUT = dataset.features_per_particle
    NUM_PARTICLES = dataset.num_particles

    input_size = NUM_PARTICLES * FEATURES_IN
    output_size = NUM_PARTICLES * FEATURES_OUT

    print(f"Detected: {NUM_PARTICLES*2} particles, {FEATURES_IN} input / {FEATURES_OUT} output features per particle")
    print(f"Using {len(dataset)} samples (max_frames={max_frames}), split: {train_size} train / {val_size} val")

    model = ParticleNet(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                pred_val = model(x_val)
                loss = loss_fn(pred_val, y_val)
                val_loss += loss.item()

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ptc_model_best_state.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    print("Training done. Loading best model.")
    model.load_state_dict(torch.load("ptc_model_best_state.pt"))
    torch.save(model, "ptc_model_full.pt")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train ParticleNet on simulation data in SQLite DB.")
    parser.add_argument("--db_path", type=str, default="./dataset.db", help="SQLite database path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_frames", type=int, default=None, help="Max number of frames to load from DB (default: all)")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Patience for early stopping")

    args = parser.parse_args()
    train(
        args.db_path,
        max_frames=args.max_frames,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_patience=args.early_stop_patience
    )

if __name__ == "__main__":
    main()
