import json
import sys
import os
from tqdm import tqdm

input_args = sys.argv[1:-1]  # Firstly list all the seperate datasets
output_file = sys.argv[-1]  # Lastly name the output path

input_files = []

if len(input_args) == 1 and os.path.isdir(input_args[0]):
    folder = input_args[0]
    for file in sorted(os.listdir(folder)):
        if file.endswith(".json"):
            input_files.append(os.path.join(folder, file))
else:
    for arg in input_args:
        input_files.append(arg)

merged_data = []
input_files = list(set(input_files))
input_files.sort()
print(f"{len(input_files)} Files found.")

i = 1
for path in input_files:
    print(f"f{i}: Load {path} ..")
    i += 1
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

for path in input_files:
    os.remove(path)

print("Merge complete.")
print("Garbage collecting .. ")
