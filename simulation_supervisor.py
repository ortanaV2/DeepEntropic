import subprocess
import time
from datetime import datetime

def run_parallel_batch(batch_size, batch_num, total_batches):
    print("="*60)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Starting batch {batch_num} of {total_batches} with {batch_size} parallel simulations")
    print("="*60)

    start_time = time.time()

    procs = []
    for i in range(batch_size):
        print(f"  -> Starting simulation {i+1} in batch {batch_num}...")
        p = subprocess.Popen(["./simulation.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((i+1, p))

    for i, p in procs:
        stdout, stderr = p.communicate()
        duration = time.time() - start_time
        print(f"\n[Batch {batch_num} | Simulation {i}] Finished with exit code: {p.returncode}")
        print(f"Duration since batch start: {duration:.2f} seconds")
        if stdout.strip():
            print(f"Simulation {i} output:\n{stdout.strip()}")
        if stderr.strip():
            print(f"Simulation {i} error output:\n{stderr.strip()}")
        print("-"*40)

    total_duration = time.time() - start_time
    print(f"[Batch {batch_num}] All simulations finished in {total_duration:.2f} seconds\n")

def run_all_batches(total_batches, batch_size, delay_between_batches=0.25):
    for batch_num in range(1, total_batches + 1):
        run_parallel_batch(batch_size, batch_num, total_batches)
        if batch_num < total_batches:
            time.sleep(delay_between_batches)

if __name__ == "__main__":
    total_batches = 9000
    batch_size = 5
    run_all_batches(total_batches, batch_size)
