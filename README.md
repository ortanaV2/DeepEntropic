<img width="1920" height="575" alt="Deep_Entropic_Logo_Crop" src="https://github.com/user-attachments/assets/97c727dc-d3e9-4d1d-9af9-99a51cd551ca" />

#### Registered Project for the Austrian Society for Artificial Intelligence (ASAI) Competition (BWKI): Accelerating Physical Simulations Using Neural Networks.

This project explores the use of neural networks to significantly improve the speed and performance of physical simulations, particularly in Python and related environments.
At its core, DeepEntropic is a hybrid simulation and machine learning framework for modeling and predicting the dynamics of interacting particles. It combines a high-performance C-based physics engine with structured data collection via SQLite and neural network training using PyTorch. The goal is to replace or accelerate traditional simulations by learning physical behaviors directly from data.

## Features

- **Particle Simulation:** Fast C/SDL2-based simulation of particle systems with gravity, pressure, viscosity, and boundary effects.
- **Data Collection:** Automatic recording of simulation frames into an SQLite database for ML training.
- **Parallel Simulation:** Supervisor script for running many simulations in parallel batches.
- **Neural Network Training:** PyTorch code for training a model to predict particle movement from simulation data.
- **Benchmarking & Visualization:** Python script for visualizing and benchmarking trained models.

## Project Structure

```
DeepEntropic/
├── simulation.c                # Particle simulation (C, SDL2, SQLite)
├── simulation_supervisor.py    # Parallel batch runner for simulation
├── model_training.py           # PyTorch training script
├── model_benchmark.py          # Model evaluation & visualization
├── dataset.db                  # SQLite database (generated)
├── ptc_model_full.pt           # Trained model (generated)
```

## Requirements

### Simulation (C)

- SDL2
- SQLite3
- OpenMP (optional, for parallel force computation)
- GCC/Clang (Linux/macOS) or MSVC (Windows)

### Python

- Python 3.10
- PyTorch
- numpy
- matplotlib

## Usage

### 1. Build the Simulation

On Linux/macOS:
```sh
gcc -fopenmp simulation.c -o simulation `sdl2-config --cflags --libs` -lsqlite3 -O2 -lm
```

On Windows (MinGW/MSYS2):
```sh
gcc -fopenmp simulation.c -o simulation `sdl2-config --cflags --libs` -lsqlite3 -O2 -lm
```

### 2. Run Simulations

You can run a single simulation:
```sh
./simulation.exe
```
Or use the supervisor to run many in parallel:
```sh
python simulation_supervisor.py
```
This will generate or append to `dataset.db`.

### 3. Train the Model

```sh
python model_training.py --db_path dataset.db --epochs 50 --batch_size 64
```
This will produce `ptc_model_full.pt`.

### 4. Benchmark & Visualize

```sh
python model_benchmark.py
```
This will visualize the predicted particle trajectories using the trained model.

## Data Format

- **Inputs:** For each frame, each particle has 4 features: `prev_x`, `prev_y`, `cur_x`, `cur_y` (normalized to [0,1]).
- **Targets:** For each frame, each particle has 2 features: `dx`, `dy` (change in normalized position).

## Customization

- Change simulation parameters in `simulation.c` (number of particles, physics constants, etc.).
- Adjust neural network architecture in `model_training.py` and `model_benchmark.py`.

## Troubleshooting

- If you encounter multiprocessing/pickling errors in PyTorch DataLoader, set `num_workers=0`.
- Make sure SDL2 and SQLite3 are installed and available to your compiler.
- For large-scale simulation, ensure sufficient disk space for `dataset.db`.

## License

[MIT License](https://github.com/ortanaV2/DeepEntropic/blob/main/LICENSE)
