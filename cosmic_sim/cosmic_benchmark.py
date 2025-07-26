import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

WIDTH = 800
HEIGHT = 600
NUM_PARTICLES = 2000
RADIUS = 4
DIAMETER = RADIUS * 2

FRAME_TIME = 0.016  # 16 ms in seconds
RECORD_SECONDS = 300
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

INPUT_SIZE = NUM_PARTICLES * 4 + 1  # prev_x,y + curr_x,y + framecount (normiert)
HIDDEN_SIZE = 512
OUTPUT_SIZE = NUM_PARTICLES * 2  # next positions (x,y)

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
    half = NUM_PARTICLES // 2
    positions[:half, 0] = WIDTH / 2 - 200 + np.random.uniform(-100, 100, half)
    positions[:half, 1] = HEIGHT / 2 + np.random.uniform(0, 100, half)
    positions[half:, 0] = WIDTH / 2 + 200 + np.random.uniform(-100, 100, NUM_PARTICLES - half)
    positions[half:, 1] = HEIGHT / 2 + np.random.uniform(0, 100, NUM_PARTICLES - half)

    prev_positions = positions.copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=RADIUS*4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')

    plt.pause(3)

    for frame in range(TOTAL_FRAMES):
        inp_np = np.hstack((prev_positions, positions)).astype(np.float32).flatten()
        inp_np = positions.astype(np.float32).flatten()  #! Dirty fix (only use current position)

        inp = torch.tensor(inp_np, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(inp)

        next_positions = out.cpu().numpy().reshape(NUM_PARTICLES, 2)
        next_positions = clamp_positions(next_positions)

        prev_positions = positions.copy()
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
