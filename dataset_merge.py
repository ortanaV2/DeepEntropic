import json
import sys
from tqdm import tqdm

input_files = sys.argv[1:-1]  # Firstly list all the seperate datasets
output_file = sys.argv[-1]  # Lastly name the output path

merged_data = []

for path in input_files:
    print(f"Load {path} ..")
    with open(path, "r") as f:
        data = json.load(f)
        for item in tqdm(data, desc=f"Appending {path}", unit="frames"):
            merged_data.append(item)

with open(output_file, "w") as f_out:
    f_out.write("[\n")
    for i, item in enumerate(tqdm(merged_data, desc="Saving", unit="frames")):
        json.dump(item, f_out)
        f_out.write(",\n" if i < len(merged_data) - 1 else "\n")
    f_out.write("]")

print("Merge complete.")
