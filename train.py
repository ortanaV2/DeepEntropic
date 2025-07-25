import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ParticleDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        self.targets = torch.tensor(data['targets'], dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ParticleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def train(json_path, input_size, hidden_size, output_size, epochs=50, batch_size=32, lr=0.001):
    dataset = ParticleDataset(json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ParticleNet(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

    return model

if __name__ == "__main__":
    model = train("particle_data.json", input_size=14000, hidden_size=512, output_size=14000)
