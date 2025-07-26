import subprocess
import shutil
import os
import sys

def run_simulations_and_merge():
    base_path = "./fluid_sim"
    exe_path = "./fluid_sim/fluid_sim.exe"
    original_output = "simulation_data.json"

    output_files = []

    for i in range(1, int(sys.argv[1])+1):  # New dataset generation amount
        print(f"Run simulation {i}...")
        result = subprocess.run([exe_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running simulation {i}:")
            print(result.stderr)
            return
        
        new_output = os.path.join(base_path, f"simulation_data_{i}.json")

        if os.path.exists(original_output):
            shutil.move(original_output, new_output)
            output_files.append(new_output)
            print(f"Renamed output to {new_output}")
        else:
            print(f"Output file not found after simulation {i}")
            return

    merge_command = ["python", "dataset_merge.py"] + output_files + [os.path.join(base_path, "merged_simulation_data.json")]
    print("Merging datasets...")
    result = subprocess.run(merge_command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error merging datasets:")
        print(result.stderr)
    else:
        print("Datasets merged successfully.")

if __name__ == "__main__":
    run_simulations_and_merge()
