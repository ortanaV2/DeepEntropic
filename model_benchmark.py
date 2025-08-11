import argparse
import time
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

WIDTH, HEIGHT = 800, 600
PARTICLE_RADIUS = 3
NUM_PARTICLES = 1000
FRAME_TIME = 0.016
RECORD_SECONDS = 60
TOTAL_FRAMES = int(RECORD_SECONDS / FRAME_TIME)

NUM_NEIGHBORS = 500
INPUT_DIM = 2006   # x,y,vx,vy + NUM_N * (dx,dy,dvx,dvy) + gx, gy
HIDDEN_DIM = 128
OUTPUT_DIM = 4   # dx, dy, dvx, dvy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def init_particles_from_clusters(num_particles=NUM_PARTICLES, cluster_radius=150, separation=400, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    half = num_particles // 2
    positions = np.zeros((num_particles, 2), dtype=np.float32)
    velocities = np.zeros((num_particles, 2), dtype=np.float32)

    min_dist = 2.0 * PARTICLE_RADIUS
    min_dist_sq = min_dist * min_dist

    initial_speed = 0

    def fill_cluster(start_idx, count, cx, cy):
        placed = 0
        attempts = 0
        max_attempts = 3000
        while placed < count:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            r = np.sqrt(rng.uniform(0.0, 1.0)) * cluster_radius
            x = cx + np.cos(angle) * r
            y = cy + np.sin(angle) * r

            # Check distance only against already placed particles in this cluster
            if placed > 0:
                dx = positions[start_idx:start_idx + placed, 0] - x
                dy = positions[start_idx:start_idx + placed, 1] - y
                if np.any(dx * dx + dy * dy < min_dist_sq):
                    attempts += 1
                    if attempts > count * max_attempts:
                        # fallback for stuck placement
                        remaining = count - placed
                        positions[start_idx + placed:start_idx + count, 0] = cx + rng.uniform(-1, 1, size=remaining)
                        positions[start_idx + placed:start_idx + count, 1] = cy + rng.uniform(-1, 1, size=remaining)
                        velocities[start_idx + placed:start_idx + count] = rng.uniform(-initial_speed, initial_speed, size=(remaining, 2))
                        break
                    continue

            positions[start_idx + placed] = (x, y)
            velocities[start_idx + placed] = rng.uniform(-initial_speed, initial_speed, size=2)
            placed += 1
            attempts = 0

    cx1 = WIDTH / 2.0 - separation / 2.0
    cx2 = WIDTH / 2.0 + separation / 2.0
    cy = HEIGHT / 2.0

    fill_cluster(0, half, cx1, cy)
    fill_cluster(half, num_particles - half, cx2, cy)

    return positions, velocities

def build_input_numpy(positions, velocities):
    """
    Build input features including normalized particle states,
    500 nearest neighbor deltas and global gravity (gx, gy).
    """
    N = positions.shape[0]

    # Normalize particle positions and velocities
    x_norm = (positions[:, 0] / WIDTH).astype(np.float32)
    y_norm = (positions[:, 1] / HEIGHT).astype(np.float32)
    vx_norm = (velocities[:, 0] / WIDTH).astype(np.float32)
    vy_norm = (velocities[:, 1] / HEIGHT).astype(np.float32)

    # Find 500 nearest neighbors per particle (k=501 because first result is the particle itself)
    tree = cKDTree(positions)
    _, idxs = tree.query(positions, k=NUM_NEIGHBORS + 1, workers=-1)
    nbr_idx = idxs[:, 1:NUM_NEIGHBORS + 1]  # Skip first column (self)

    nbr_pos = positions[nbr_idx]
    nbr_vel = velocities[nbr_idx]

    pos_exp = positions[:, None, :]  # Shape: (N, 1, 2)
    vel_exp = velocities[:, None, :]  # Shape: (N, 1, 2)

    rel_pos = nbr_pos - pos_exp  # Shape: (N, NUM_NEIGHBORS, 2)
    rel_vel = nbr_vel - vel_exp  # Shape: (N, NUM_NEIGHBORS, 2)

    # Normalize relative positions and velocities
    rel_pos[..., 0] /= WIDTH
    rel_pos[..., 1] /= HEIGHT
    rel_vel[..., 0] /= WIDTH
    rel_vel[..., 1] /= HEIGHT

    # Global gravity (fixed constants, normalized by screen size for scale)
    gx = 0.0
    gy = 0.1  # example gravity downward

    # Build output array
    out = np.empty((N, INPUT_DIM), dtype=np.float32)
    
    # First 4 features: particle state
    out[:, 0:4] = np.column_stack([x_norm, y_norm, vx_norm, vy_norm])

    # Next 80 features: 500 neighbors × 4 values each
    base = 4
    for i in range(NUM_NEIGHBORS):
        out[:, base + i*4 : base + i*4 + 4] = np.column_stack([
            rel_pos[:, i, 0], rel_pos[:, i, 1],
            rel_vel[:, i, 0], rel_vel[:, i, 1]
        ])

    # Last 2 features: global gravity
    out[:, -2] = gx
    out[:, -1] = gy

    return out

def run_benchmark(args):
    model = SimpleMLP(input_dim=INPUT_DIM, hidden_dim=args.hidden_dim, output_dim=OUTPUT_DIM, dropout=0.05)
    ckpt = torch.load(args.model_path, map_location=device)

    if isinstance(ckpt, dict) and set(ckpt.keys()).issubset(set(model.state_dict().keys())):
        model.load_state_dict(ckpt)
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        try:
            model = ckpt
        except Exception:
            model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    runner = model
    if args.use_jit:
        try:
            runner = torch.jit.script(model.eval()).to(device)
            print("Using TorchScript compiled model for inference.")
        except Exception as e:
            print("TorchScript compilation failed, using eager model. Error:", e)

    rng = np.random.default_rng(seed=args.seed)
    positions, velocities = init_particles_from_clusters(
        num_particles=args.num_particles,
        cluster_radius=args.cluster_radius,
        separation=args.separation,
        rng=rng
    )

    num = positions.shape[0]

    do_vis = args.visualize and (plt is not None)
    if do_vis:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=PARTICLE_RADIUS * 4, c='cyan', edgecolors='b')
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.invert_yaxis()
        ax.set_aspect('equal')

    t_total_start = time.time()
    frame_times = []

    print(f"Starting benchmark with {NUM_NEIGHBORS} neighbors per particle (INPUT_DIM = {INPUT_DIM})")

    for frame in range(args.total_frames):
        t0 = time.time()

        if not np.all(np.isfinite(positions)):
            raise ValueError("Positions contain NaN oder Inf")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("Velocities contain NaN oder Inf")

        x_np = build_input_numpy(positions, velocities)

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_np).to(device)
            out = runner(x_tensor)
            out_np = out.cpu().numpy()

        # Update normalized positions with predicted dx, dy
        max_dv = 0.1
        out_np[:, 2] = np.clip(out_np[:, 2], -max_dv, max_dv)
        out_np[:, 3] = np.clip(out_np[:, 3], -max_dv, max_dv)

        norm_pos = np.empty((num, 2), dtype=np.float32)
        norm_pos[:, 0] = positions[:, 0] / WIDTH
        norm_pos[:, 1] = positions[:, 1] / HEIGHT

        norm_pos[:, 0] += out_np[:, 0]
        norm_pos[:, 1] += out_np[:, 1]

        eps_x = PARTICLE_RADIUS / WIDTH
        eps_y = PARTICLE_RADIUS / HEIGHT
        np.clip(norm_pos[:, 0], eps_x, 1.0 - eps_x, out=norm_pos[:, 0])
        np.clip(norm_pos[:, 1], eps_y, 1.0 - eps_y, out=norm_pos[:, 1])

        # Compute denormalized positions and velocities (using predicted dvx, dvy)
        new_positions_x = norm_pos[:, 0] * WIDTH
        new_positions_y = norm_pos[:, 1] * HEIGHT

        # Velocity updates from predicted dvx, dvy instead of finite difference
        velocities[:, 0] += out_np[:, 2] * WIDTH
        velocities[:, 1] += out_np[:, 3] * HEIGHT

        positions[:, 0] = new_positions_x
        positions[:, 1] = new_positions_y

        if args.use_boundaries:
            mask_left = positions[:, 0] < PARTICLE_RADIUS
            mask_right = positions[:, 0] > (WIDTH - PARTICLE_RADIUS)
            mask_top = positions[:, 1] < PARTICLE_RADIUS
            mask_bottom = positions[:, 1] > (HEIGHT - PARTICLE_RADIUS)

            if mask_left.any():
                positions[mask_left, 0] = PARTICLE_RADIUS
                velocities[mask_left, 0] *= -args.boundary_damping
            if mask_right.any():
                positions[mask_right, 0] = WIDTH - PARTICLE_RADIUS
                velocities[mask_right, 0] *= -args.boundary_damping
            if mask_top.any():
                positions[mask_top, 1] = PARTICLE_RADIUS
                velocities[mask_top, 1] *= -args.boundary_damping
            if mask_bottom.any():
                positions[mask_bottom, 1] = HEIGHT - PARTICLE_RADIUS
                velocities[mask_bottom, 1] *= -args.boundary_damping

        frame_times.append(time.time() - t0)

        if do_vis and (frame % args.vis_every == 0):
            scatter.set_offsets(positions)
            ax.set_title(f"Frame {frame + 1}/{args.total_frames} | avg frame time {np.mean(frame_times[-50:]):.4f}s")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # Print progress every 100 frames
        if frame % 100 == 0 and frame > 0:
            avg_time_so_far = np.mean(frame_times[-100:])
            fps_so_far = 1.0 / avg_time_so_far if avg_time_so_far > 0 else 0.0
            print(f"Frame {frame}: avg frame time {avg_time_so_far:.6f}s, approx FPS: {fps_so_far:.2f}")

    total_time = time.time() - t_total_start
    avg_frame_time = np.mean(frame_times) if frame_times else 0.0
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    print("Benchmark finished.")
    print(f"Total frames: {args.total_frames}, Total time: {total_time:.4f}s")
    print(f"Avg frame time: {avg_frame_time:.6f}s, Approx FPS: {fps:.2f}")
    print(f"Input dimension used: {INPUT_DIM} (4 particle + {NUM_NEIGHBORS}*4 neighbors + 2 gravity)")

    if do_vis:
        plt.ioff()
        plt.show(block=True)

def parse_args():
    p = argparse.ArgumentParser(description="Optimized particle inference benchmark with 500 neighbors and gravity feature.")
    p.add_argument("--model_path", type=str, default="best_mlp_model.pt", help="Path to saved model")
    p.add_argument("--num_particles", type=int, default=NUM_PARTICLES)
    p.add_argument("--total_frames", type=int, default=TOTAL_FRAMES)
    p.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--use_jit", action="store_true", help="Use TorchScript compilation for faster inference")
    p.add_argument("--visualize", type=int, default=0, help="Enable visualization (0/1)")
    p.add_argument("--vis_every", type=int, default=1, help="Update visualization every N frames")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster_radius", type=float, default=150.0)
    p.add_argument("--separation", type=float, default=400.0)
    p.add_argument("--use_boundaries", type=int, default=1, help="Apply boundary bouncing")
    p.add_argument("--boundary_damping", type=float, default=0.2)
    args = p.parse_args()
    args.visualize = bool(args.visualize)
    args.use_boundaries = bool(args.use_boundaries)
    return args

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)