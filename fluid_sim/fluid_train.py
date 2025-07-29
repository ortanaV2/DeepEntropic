import argparse
import json
import sqlite3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SQLiteParticleDataset(Dataset):
    def __init__(self, db_path, chunk_size=3500):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.chunk_size = chunk_size

        self.cursor.execute("SELECT COUNT(*) FROM drop_10p_dataset")
        self.total_samples = self.cursor.fetchone()[0]

        self.features_per_particle = 4
        self.buffer_start = -1
        self.buffer = []

        self._load_chunk(0)
        self.num_particles = len(self.buffer[0][0]) // self.features_per_particle if self.buffer else 0

    def _load_chunk(self, start_idx):
        self.buffer_start = start_idx
        self.buffer = []

        query = f"SELECT inputs, targets FROM drop_10p_dataset LIMIT {self.chunk_size} OFFSET {start_idx}"
        rows = self.cursor.execute(query).fetchall()

        for row in rows:
            inputs_blob = row[0]
            targets_blob = row[1]

            inputs = list(torch.frombuffer(inputs_blob, dtype=torch.float32).numpy())
            targets = list(torch.frombuffer(targets_blob, dtype=torch.float32).numpy())

            filtered_input = []
            filtered_target = []

            for i in range(0, len(inputs), 4):
                filtered_input.extend(inputs[i:i+4])
                filtered_target.extend(targets[i:i+4])

            self.buffer.append((filtered_input, filtered_target))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if not (self.buffer_start <= idx < self.buffer_start + self.chunk_size):
            self._load_chunk(idx)
        local_idx = idx - self.buffer_start
        x, y = self.buffer[local_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

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

def train(db_path, hidden_size=512, epochs=50, batch_size=64, lr=1e-4):
    dataset = SQLiteParticleDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    FEATURES_IN = dataset.features_per_particle
    FEATURES_OUT = dataset.features_per_particle
    NUM_PARTICLES = dataset.num_particles

    input_size = NUM_PARTICLES * FEATURES_IN
    output_size = NUM_PARTICLES * FEATURES_OUT

    print(f"Detected: {NUM_PARTICLES*2} particles, {FEATURES_IN} input / {FEATURES_OUT} output features per particle")

    model = ParticleNet(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        scheduler.step()

    torch.save(model, "particle_model_full.pt")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train ParticleNet on simulation data in SQLite DB.")
    parser.add_argument("--db_path", type=str, default="./dataset.db", help="SQLite database path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train(args.db_path, hidden_size=args.hidden_size, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__":
    main()
