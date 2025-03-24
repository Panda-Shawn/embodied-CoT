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
    parser.add_argument("--original_resolution", type=int, default=256)
    parser.add_argument("--target_resolution", type=int, default=224)
    args = parser.parse_args()

    reasoning_file_path = os.path.join(args.reasoning_dir, f"reasoning.json")
    with open(reasoning_file_path, "r") as f:
        reasonings = json.load(f)

    if args.results_path is None:
        args.results_path = os.path.join(args.reasoning_dir, f"reasoning_{args.target_resolution}.json")

    scale = args.target_resolution / args.original_resolution

    for file_path in tqdm(reasonings.keys(), desc="Scaling"):
        bbox = reasonings[file_path]["0"]["features"]["bboxes"]
        gripper_position = reasonings[file_path]["0"]["features"]["gripper_position"]

        for step_idx in range(len(bbox)):
            for i in range(len(bbox[step_idx])):
                # import pdb; pdb.set_trace()
                bbox[step_idx][i][1] = [int(x * scale) for x in bbox[step_idx][i][1]]
            gripper_position[step_idx] = [int(x * scale) for x in gripper_position[step_idx]]

    with open(args.results_path, "w") as f:
        json.dump(reasonings, f)

