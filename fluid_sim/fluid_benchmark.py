import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

WIDTH = 1200
HEIGHT = 1000
RADIUS = 6
DIAMETER = RADIUS * 2

FRAME_TIME = 0.016  # 16 ms per frame
RECORD_SECONDS = 10
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

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

def init_centered_particles(num_particles, radius, width, height):
    spawn_width = 240
    spawn_height = 120
    spawn_x_min = (width / 2.0) - (spawn_width / 2.0)
    spawn_y_min = (height / 2.0) - (spawn_height / 2.0)
    min_dist = 2.5 * radius
    max_attempts = 1000

    positions = []
    for i in range(num_particles):
        for attempt in range(max_attempts):
            x = np.random.uniform(spawn_x_min, spawn_x_min + spawn_width)
            y = np.random.uniform(spawn_y_min, spawn_y_min + spawn_height)
            valid = True
            for px, py in positions:
                if (x - px)**2 + (y - py)**2 < min_dist**2:
                    valid = False
                    break
            if valid:
                positions.append((x, y))
                break
        else:
            positions.append((x, y))  # Place anyway if no valid found

    return np.array(positions, dtype=np.float32)

def normalize_positions(positions):
    norm_x = positions[:, 0] / WIDTH
    norm_y = positions[:, 1] / HEIGHT
    return np.stack([norm_x, norm_y], axis=1)

def denormalize_positions(norm_positions):
    x = norm_positions[:, 0] * WIDTH
    y = norm_positions[:, 1] * HEIGHT
    return np.stack([x, y], axis=1)

def main():
    model = torch.load("particle_model_full.pt", map_location=device)
    model.eval()

    input_size = model.net[0].in_features
    NUM_PARTICLES = input_size // 2

    positions = init_centered_particles(NUM_PARTICLES, RADIUS, WIDTH, HEIGHT)
    velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)

    norm_positions = normalize_positions(positions)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=RADIUS*4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for frame in range(TOTAL_FRAMES):
        input_data = norm_positions.flatten()
        inp = torch.tensor(input_data, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            delta = model(inp).cpu().numpy().reshape(NUM_PARTICLES, 2)

        norm_positions += delta

        norm_positions = np.clip(norm_positions, [RADIUS/WIDTH, RADIUS/HEIGHT], [1 - RADIUS/WIDTH, 1 - RADIUS/HEIGHT])

        positions = denormalize_positions(norm_positions)
        positions = clamp_positions(positions)
        norm_positions = normalize_positions(positions)

        scatter.set_offsets(positions)
        ax.set_title(f"Frame {frame+1}/{TOTAL_FRAMES}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(FRAME_TIME)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
