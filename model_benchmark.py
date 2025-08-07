import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Constants
WIDTH, HEIGHT = 800, 600
PARTICLE_RADIUS = 3
NUM_PARTICLES = 1000
FRAME_TIME = 0.016
RECORD_SECONDS = 12
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 5  # x, y, vx, vy, nnd (normalized distance to nearest neighbor)
HIDDEN_DIM = 128
OUTPUT_DIM = 4  # dx, dy, vx, vy

# Simple MLP Model (replaces the GNN)
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)

def init_particles_from_clusters(num_particles=NUM_PARTICLES, cluster_radius=150, separation=400):
    half = num_particles // 2
    min_dist = 2.0 * PARTICLE_RADIUS
    clusters = [
        {"cx": WIDTH / 2 - separation / 2, "cy": HEIGHT / 2, "vy": 0, "start": 0, "end": half},
        {"cx": WIDTH / 2 + separation / 2, "cy": HEIGHT / 2, "vy": 0, "start": half, "end": num_particles}
    ]
    positions = np.zeros((num_particles, 2), dtype=np.float32)
    velocities = np.zeros((num_particles, 2), dtype=np.float32)

    def is_valid(start_idx, end_idx, x, y):
        for j in range(start_idx, end_idx):
            dx = positions[j, 0] - x
            dy = positions[j, 1] - y
            if dx * dx + dy * dy < min_dist * min_dist:
                return False
        return True

    max_attempts = 3000
    for cluster in clusters:
        count = cluster["end"] - cluster["start"]
        cx, cy = cluster["cx"], cluster["cy"]
        start_idx = cluster["start"]
        for i in range(count):
            attempts = 0
            while attempts < max_attempts:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1)) * cluster_radius
                x = cx + np.cos(angle) * r
                y = cy + np.sin(angle) * r
                if is_valid(start_idx, start_idx + i, x, y):
                    positions[start_idx + i] = [x, y]
                    velocities[start_idx + i] = [0.0, cluster["vy"]]
                    break
                attempts += 1
            if attempts == max_attempts:
                positions[start_idx + i] = [cx, cy]
                velocities[start_idx + i] = np.random.uniform(-0.5, 0.5, size=2)  # init vel

    return positions, velocities

# Normalization helpers
def normalize_positions(pos): return np.stack([pos[:, 0] / WIDTH, pos[:, 1] / HEIGHT], axis=1)
def denormalize_positions(norm): return np.stack([norm[:, 0] * WIDTH, norm[:, 1] * HEIGHT], axis=1)

# Calculate normalized distance to nearest neighbor
def compute_normalized_nn_distances(positions):
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)  # k=2 because 1st nearest is the point itself
    nearest_dists = dists[:, 1]  # second column: nearest other point
    return nearest_dists / WIDTH  # Normalize to [0, 1] by WIDTH (same as im Training)

# Prepare input tensor for MLP
def build_input_tensor(pos, vel):
    norm_pos = normalize_positions(pos)
    cf_norm = compute_normalized_nn_distances(pos)
    x = np.concatenate([norm_pos, vel, cf_norm[:, None]], axis=1)
    return torch.tensor(x, dtype=torch.float32, device=device)

# Main simulation
def main():
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load("best_mlp_model.pt", map_location=device))
    model.eval()

    cur_pos, cur_vel = init_particles_from_clusters()

    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(cur_pos[:, 0], cur_pos[:, 1], s=PARTICLE_RADIUS * 4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for frame in range(TOTAL_FRAMES):
        x_tensor = build_input_tensor(cur_pos, cur_vel)
        with torch.no_grad():
            out = model(x_tensor).cpu().numpy()

        delta_pos = out[:, :2]
        cur_vel = out[:, 2:]

        norm_cur = normalize_positions(cur_pos)
        norm_cur += delta_pos
        norm_cur = np.clip(norm_cur,
                           [PARTICLE_RADIUS / WIDTH, PARTICLE_RADIUS / HEIGHT],
                           [1 - PARTICLE_RADIUS / WIDTH, 1 - PARTICLE_RADIUS / HEIGHT])
        cur_pos = denormalize_positions(norm_cur)
        cur_pos[:, 0] = np.clip(cur_pos[:, 0], PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        cur_pos[:, 1] = np.clip(cur_pos[:, 1], PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)

        scatter.set_offsets(cur_pos)
        ax.set_title(f"Frame {frame + 1}/{TOTAL_FRAMES}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
