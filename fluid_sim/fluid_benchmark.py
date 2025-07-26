import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

WIDTH = 800
HEIGHT = 600
NUM_PARTICLES = int(sys.argv[1])
RADIUS = 6
DIAMETER = RADIUS * 2

FRAME_TIME = 1  # 16 ms per frame
RECORD_SECONDS = 10
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

INPUT_SIZE = NUM_PARTICLES * 4  # x, y, vx, vy per particle
HIDDEN_SIZE = 512
OUTPUT_SIZE = NUM_PARTICLES * 4  # Δx, Δy, Δvx, Δvy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def clamp_positions(positions):
    positions[:, 0] = np.clip(positions[:, 0], RADIUS, WIDTH - RADIUS)
    positions[:, 1] = np.clip(positions[:, 1], RADIUS, HEIGHT - RADIUS)
    return positions

def main():
    model = torch.load("particle_model_full.pt", map_location=device)
    model.eval()

    positions = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
    velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)

    positions[:, 0] = np.random.uniform(300, 500, NUM_PARTICLES)
    positions[:, 1] = np.random.uniform(50, 150, NUM_PARTICLES)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=RADIUS*4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')

    for frame in range(TOTAL_FRAMES):
        input_data = np.hstack((positions, velocities)).flatten()
        inp = torch.tensor(input_data, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(inp).cpu().numpy().reshape(NUM_PARTICLES, 4)

        delta_pos = out[:, 0:2]
        delta_vel = out[:, 2:4]

        velocities += delta_vel
        positions += delta_pos
        positions = clamp_positions(positions)

        scatter.set_offsets(positions)
        ax.set_title(f"Frame {frame+1}/{TOTAL_FRAMES}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(FRAME_TIME)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
