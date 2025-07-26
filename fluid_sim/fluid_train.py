import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class ParticleDataset(Dataset):
    def __init__(self, json_path, input_mode="xyd"):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.input_mode = input_mode
        self.samples = []

        for entry in data:
            full_input = entry['inputs']
            full_target = entry['targets']

            if input_mode == "xy":
                features_per_particle = 2
                filtered_input = []
                for i in range(0, len(full_input), 4):
                    filtered_input.extend(full_input[i:i+2])
                filtered_target = []
                for i in range(0, len(full_target), 4):
                    filtered_target.extend(full_target[i:i+2])
            elif input_mode == "xyd":
                features_per_particle = 4
                filtered_input = []
                for i in range(0, len(full_input), 4):
                    filtered_input.extend(full_input[i:i+4])
                filtered_target = []
                for i in range(0, len(full_target), 4):
                    filtered_target.extend(full_target[i:i+4])
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            self.samples.append((filtered_input, filtered_target))

        self.inputs = torch.tensor([s[0] for s in self.samples], dtype=torch.float32)
        self.targets = torch.tensor([s[1] for s in self.samples], dtype=torch.float32)

        self.features_per_particle = features_per_particle
        self.num_particles = len(self.inputs[0]) // features_per_particle

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

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

def train(json_path, input_mode="xyd", hidden_size=512, epochs=50, batch_size=64, lr=1e-4):
    dataset = ParticleDataset(json_path, input_mode=input_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    FEATURES_IN = dataset.features_per_particle
    FEATURES_OUT = dataset.features_per_particle
    NUM_PARTICLES = dataset.num_particles

    input_size = NUM_PARTICLES * FEATURES_IN
    output_size = NUM_PARTICLES * FEATURES_OUT

    print(f"Detected: {NUM_PARTICLES} particles, {FEATURES_IN} input / {FEATURES_OUT} output features per particle")

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

    torch.save(model, f"particle_model_full.pt")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train ParticleNet on simulation data.")
    parser.add_argument("--dataset_path", type=str, default="./simulation_data.json", help="Simulation data path (dataset)")
    parser.add_argument("--input_mode", type=str, default="xyd", choices=["xy", "xyd"], help="dataset configuration: xy or xyd")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")

    args = parser.parse_args()
    train(args.dataset_path, input_mode=args.input_mode, epochs=args.epochs)

if __name__ == "__main__":
    main()
