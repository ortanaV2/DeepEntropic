import os 
import argparse

p = argparse.ArgumentParser()
p.add_argument("--table", type=str, default="debug", help="dataset table")
p.add_argument("--batch_size", type=int, default="1", help="Simulation instances simultaneously")
p.add_argument("--batches", type=int, default="200", help="Simulation runs recorded")
p.add_argument("--epochs", type=int, default="20", help="Trainig epochs")
p.add_argument("--limit", type=int, default="2000", help="Amount of frames used for training")
args = p.parse_args()

print("[1] Task: Build Simulation Dataset.")
os.system(f"python simulation_supervisor.py --table {args.table} --batch_size {args.batch_size} --batches {args.batches}")
print("[2] Task: Train Model.")
os.system(f"python model_training.py --table {args.table} --epochs {args.epochs} --limit {args.limit}")
print("Task chain [1],[2] finished.")
