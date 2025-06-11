import json
import math
import os
from pathlib import Path

def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1["x"] - p2[0])**2 +
        (p1["y"] - p2[1])**2
    )

def main():
    out_file = Path("./error_distances.txt")

    with open(out_file, "w") as fout:
        fout.write("file,iter,avg_error\n")
        for idx in range(1, 31):
            proc_path = Path(f"./{idx}/processed_can_data_{idx}.txt")
            dest_path = Path(f"./{idx}/can_dest_{idx}.txt")

            if not proc_path.exists() or not dest_path.exists():
                print(f"skipping {idx}: missing file")
                continue

            proc_data = json.load(proc_path.open())
            dest_data = json.load(dest_path.open())

            dest_coords = proc_data.get("destination_coordinates", [])

            for i in range(10):
                key = f"iter{i}"
                iter_list = dest_data.get(key)
                if not iter_list or len(iter_list) != len(dest_coords):
                    continue

                distances = [
                    euclidean_distance(dest_coords[j], iter_list[j])
                    for j in range(len(dest_coords))
                ]
                for k in range(len(distances)):
                    fout.write(f"{idx},{key},{distances[k]:.6f}\n")

    print(f"results saved to {out_file}")

if __name__ == "__main__":
    main()
