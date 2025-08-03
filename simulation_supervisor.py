import subprocess
import time
from datetime import datetime
import argparse

def run_parallel_batch(batch_size, batch_num, total_batches, table_name):
    print("="*60)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Starting batch {batch_num} of {total_batches} with {batch_size} parallel simulations")
    print("="*60)

    start_time = time.time()
    procs = []

    for i in range(batch_size):
        print(f"  -> Starting simulation {i+1} in batch {batch_num}...")
        time.sleep(0.05)
        p = subprocess.Popen(
            ["./simulation.exe", table_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
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

def run_all_batches(total_batches, batch_size, table_name, delay_between_batches=0.25):
    for batch_num in range(1, total_batches + 1):
        run_parallel_batch(batch_size, batch_num, total_batches, table_name)
        if batch_num < total_batches:
            time.sleep(delay_between_batches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="Name of the SQL table to write into")
    parser.add_argument("--batches", type=int, default=10000, help="Number of total batches")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of simulations per batch")
    args = parser.parse_args()

    run_all_batches(args.batches, args.batch_size, args.table)
