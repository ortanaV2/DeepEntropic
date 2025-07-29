import subprocess
import time

def run_parallel_batch(batch_size):
    procs = []
    for i in range(batch_size):
        print(f"Starte Simulation {i+1} im Batch...")
        p = subprocess.Popen(["./fluid_sim.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((i+1, p))
    
    for i, p in procs:
        stdout, stderr = p.communicate()
        print(f"Simulation {i} beendet mit Exit-Code {p.returncode}")
        if stdout:
            print(f"Output Simulation {i}:\n{stdout}")
        if stderr:
            print(f"Fehler Simulation {i}:\n{stderr}")
        print("-" * 40)

def run_all_batches(total_batches, batch_size, delay_between_batches=0.25):
    for batch_num in range(total_batches):
        print(f"Starte Batch {batch_num+1} von {total_batches}")
        run_parallel_batch(batch_size)
        if batch_num < total_batches - 1:
            time.sleep(delay_between_batches)

if __name__ == "__main__":
    total_batches = 10000
    batch_size = 5
    run_all_batches(total_batches, batch_size)
