import json

import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_dir", type=str, default="/data/lzx/embodied-CoT/scripts/generate_embodied_data/new_reasonings/final_reasonings")
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    reasoning_file_path = os.path.join(args.reasoning_dir, f"reasoning.json")
    with open(reasoning_file_path, "r") as f:
        reasonings = json.load(f)

    if args.results_path is None:
        args.results_path = os.path.join(args.reasoning_dir, f"reasoning_primitives.json")

    for file_path in tqdm(reasonings.keys(), desc="Scaling"):
        ori_primitive = reasonings[file_path]["0"]["features"]["move_primitive"]

        primitive = ori_primitive[file_path]

        reasonings[file_path]["0"]["features"]["move_primitive"] = primitive

    with open(args.results_path, "w") as f:
        json.dump(reasonings, f)

