import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

WIDTH = 800
HEIGHT = 600
NUM_PARTICLES = 2000
RADIUS = 6
DIAMETER = RADIUS * 2

GRAVITY = 0.2
PRESSURE = 0.04
VISCOSITY = 0.03
DAMPING = 0.2

FRAME_TIME = 0.016  # 16 ms in seconds
RECORD_SECONDS = 10
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

INPUT_SIZE = NUM_PARTICLES * 2  # x,y per particle
HIDDEN_SIZE = 512
OUTPUT_SIZE = NUM_PARTICLES * 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def clamp_positions(positions):
    positions[:, 0] = np.clip(positions[:, 0], RADIUS, WIDTH - RADIUS)
    positions[:, 1] = np.clip(positions[:, 1], RADIUS, HEIGHT - RADIUS)
    return positions

def main():
    model = torch.load("particle_model_full.pt", map_location=device)
    model.eval()

    positions = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
    positions[:, 0] = np.random.uniform(300, 500, NUM_PARTICLES)
    positions[:, 1] = np.random.uniform(50, 150, NUM_PARTICLES)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=RADIUS*4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')

    for frame in range(TOTAL_FRAMES):
        inp = torch.tensor(positions.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(inp)

        next_positions = out.cpu().numpy().reshape(NUM_PARTICLES, 2)

        next_positions = clamp_positions(next_positions)

        positions = next_positions

        scatter.set_offsets(positions)
        ax.set_title(f"Frame {frame+1}/{TOTAL_FRAMES}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(FRAME_TIME)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
