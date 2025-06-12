import json
import math
import os
from pathlib import Path

def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1["x"] - p2["x"])**2 +
        (p1["y"] - p2["y"])**2
    )

def main():
    out_file = Path("./detection_error.txt")

    with open(out_file, "w") as fout:
        for idx in range(1, 31):
            proc_path = Path(f"./{idx}/processed_can_data_{idx}.txt")

            if not proc_path.exists():
                print(f"skipping {idx}: missing file")
                continue

            proc_data = json.load(proc_path.open())

            for i in range(10):
                key = f"iter{i}"
                iter_list = proc_data.get(key)
                scoord = iter_list.get("starting_coordinates", [])
                scoord = sorted(scoord, key=lambda d: d["y"], reverse=True)
                sref = iter_list.get("starting_reference", [])
                sref = sorted(sref, key=lambda d: d["y"], reverse=True)


                distances = [
                    euclidean_distance(scoord[j], sref[j])
                    for j in range(len(scoord))
                ]
                for k in range(len(distances)):
                    fout.write(f"{distances[k]:.6f}\n")

    print(f"results saved to {out_file}")

if __name__ == "__main__":
    main()
