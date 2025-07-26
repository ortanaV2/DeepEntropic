import json
from tqdm import tqdm

file1 = "./cosmic_sim/simulation_data_planet.json"
file2 = "./cosmic_sim/simulation_data_random.json"
output_file = "./cosmic_sim/merged_simulation_data.json"

print("Load dataset-1 ..")
with open(file1, "r") as f1:
    data1 = json.load(f1)

print("Load dataset-2 ..")
with open(file2, "r") as f2:
    data2 = json.load(f2)

print("Merge datasets ..")
merged_data = data1.copy()
for item in tqdm(data2, desc="Merging", unit="frames"):
    merged_data.append(item)

with open(output_file, "w") as f_out:
    f_out.write("[\n")
    for i, item in enumerate(tqdm(merged_data, desc="Saving", unit="frames")):
        json.dump(item, f_out)
        f_out.write(",\n" if i < len(merged_data) - 1 else "\n")
    f_out.write("]")
print("Merge complete.")