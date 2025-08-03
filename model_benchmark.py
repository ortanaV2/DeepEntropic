import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph

# Simulation parameters
WIDTH, HEIGHT = 800, 600
PARTICLE_RADIUS = 3
NUM_PARTICLES = 2000
HALF_PARTICLES = NUM_PARTICLES // 2
GRAPH_RADIUS = 0.2
FRAME_TIME = 0.016
RECORD_SECONDS = 5
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Graph Neural Network Model Definition
INPUT_DIM = 4   # Concatenated previous and current normalized positions per node
HIDDEN_DIM = 128

class SimpleGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')
        # Node feature embedding
        self.node_mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        # Edge message function
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        # Output MLP predicts position delta (dx, dy)
        self.out_mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2)
        )

    def forward(self, x, edge_index):
        x = self.node_mlp(x)
        x = self.propagate(edge_index, x=x)
        return self.out_mlp(x)

    def message(self, x_i, x_j):
        # Edge message input: concatenation of source and target node features
        edge_input = torch.cat([x_i, x_j], dim=1)
        return self.edge_mlp(edge_input)


# Particle Initialization in Two Spatially Separated Clusters
def init_particles_from_clusters(num_particles=NUM_PARTICLES, cluster_radius=150, separation=400,
                                 width=WIDTH, height=HEIGHT, radius=PARTICLE_RADIUS):
    half = num_particles // 2
    min_dist = 2.0 * radius  # Minimum allowed distance between particles

    # Define two cluster centers spaced by 'separation'
    clusters = [
        {"cx": width / 2 - separation / 2, "cy": height / 2, "vy": 0, "start": 0, "end": half},
        {"cx": width / 2 + separation / 2, "cy": height / 2, "vy": 0, "start": half, "end": num_particles}
    ]

    positions = np.zeros((num_particles, 2), dtype=np.float32)
    velocities = np.zeros((num_particles, 2), dtype=np.float32)

    def is_position_valid(start_idx, end_idx, x, y):
        # Checks if (x,y) is at least 'min_dist' away from all already placed particles in the cluster
        for j in range(start_idx, end_idx):
            dx = positions[j, 0] - x
            dy = positions[j, 1] - y
            if dx * dx + dy * dy < min_dist * min_dist:
                return False
        return True

    max_attempts = 3000

    # Place particles randomly inside each cluster radius with rejection sampling to avoid overlaps
    for cluster in clusters:
        count = cluster["end"] - cluster["start"]
        cx, cy = cluster["cx"], cluster["cy"]
        start_idx = cluster["start"]

        for i in range(count):
            attempts = 0
            while attempts < max_attempts:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.sqrt(np.random.uniform(0, 1)) * cluster_radius  # Uniform sampling in circle
                x = cx + np.cos(angle) * r
                y = cy + np.sin(angle) * r

                if is_position_valid(start_idx, start_idx + i, x, y):
                    positions[start_idx + i] = [x, y]
                    velocities[start_idx + i] = [0.0, cluster["vy"]]
                    break
                attempts += 1

            # Fallback if valid position not found after many attempts
            if attempts == max_attempts:
                positions[start_idx + i] = [cx, cy]
                velocities[start_idx + i] = [0.0, cluster["vy"]]

    return positions, velocities


# Position Normalization Helpers
def normalize_positions(positions):
    # Scale positions to [0,1] range relative to simulation bounds
    return np.stack([positions[:, 0] / WIDTH, positions[:, 1] / HEIGHT], axis=1)

def denormalize_positions(norm_positions):
    # Restore scaled positions back to simulation coordinates
    return np.stack([norm_positions[:, 0] * WIDTH, norm_positions[:, 1] * HEIGHT], axis=1)


# Build Graph for GNN from Previous and Current Positions
def build_graph(prev_pos, cur_pos, radius=GRAPH_RADIUS):
    # Create node features by concatenating previous and current normalized positions
    prev_tensor = torch.tensor(prev_pos, dtype=torch.float32, device=device)
    cur_tensor = torch.tensor(cur_pos, dtype=torch.float32, device=device)
    x = torch.cat([prev_tensor, cur_tensor], dim=1)

    # Construct edges based on spatial proximity (radius graph)
    edge_index = radius_graph(cur_tensor, r=radius, loop=False)

    return Data(x=x, edge_index=edge_index)


# Main Simulation Loop with Model Inference and Visualization
def main():
    model = SimpleGNN().to(device)
    model.load_state_dict(torch.load("best_gnn_model.pt", map_location=device))
    model.eval()

    # Initialize two spatially separated clusters of particles
    cur_pos, velocity = init_particles_from_clusters(separation=400)
    prev_pos = cur_pos - (velocity * FRAME_TIME)

    norm_cur = normalize_positions(cur_pos)
    norm_prev = normalize_positions(prev_pos)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(cur_pos[:, 0], cur_pos[:, 1], s=PARTICLE_RADIUS * 4, c='cyan', edgecolors='b')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for frame in range(TOTAL_FRAMES):
        graph = build_graph(norm_prev, norm_cur).to(device)

        with torch.no_grad():
            delta = model(graph.x, graph.edge_index).cpu().numpy()

        # Update normalized positions with predicted displacement, enforcing boundary constraints
        new_norm_pos = norm_cur + delta
        new_norm_pos = np.clip(new_norm_pos,
                               [PARTICLE_RADIUS / WIDTH, PARTICLE_RADIUS / HEIGHT],
                               [1 - PARTICLE_RADIUS / WIDTH, 1 - PARTICLE_RADIUS / HEIGHT])

        norm_prev, norm_cur = norm_cur, new_norm_pos

        # Convert back to absolute coordinates for visualization and boundary clipping
        pos = denormalize_positions(norm_cur)
        pos[:, 0] = np.clip(pos[:, 0], PARTICLE_RADIUS, WIDTH - PARTICLE_RADIUS)
        pos[:, 1] = np.clip(pos[:, 1], PARTICLE_RADIUS, HEIGHT - PARTICLE_RADIUS)

        scatter.set_offsets(pos)
        ax.set_title(f"Frame {frame + 1}/{TOTAL_FRAMES}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
