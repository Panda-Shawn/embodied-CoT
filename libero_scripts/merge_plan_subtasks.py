import argparse
import json
from tqdm import tqdm
import os

# try:
#     hydra.initialize(config_path="../calvin/calvin_models/conf")
# except Exception:
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data/lzx/embodied-CoT/scripts/generate_embodied_data/new_reasonings")
    args = parser.parse_args()

    bboxes_file_path = os.path.join(args.data_dir, args.libero_task_suite+"_w_mask", "cot", f"{args.libero_task_suite}_bboxes.json")
    with open(bboxes_file_path, "r") as f:
        bboxes = json.load(f)

    gripper_positions_file_path = os.path.join(args.data_dir, args.libero_task_suite+"_w_mask", "cot", f"{args.libero_task_suite}_no_noops_gripper_pos.json")
    with open(gripper_positions_file_path, "r") as f:
        gripper_positions = json.load(f)

    primitives_file_path = os.path.join(args.data_dir, args.libero_task_suite+"_w_mask", "cot", f"{args.libero_task_suite}_primitives.json")
    with open(primitives_file_path, "r") as f:
        primitives = json.load(f)

    # reasonings_file_path = os.path.join(args.data_dir, args.libero_task_suite+"_w_mask", f"{args.libero_task_suite}_plan_subtasks.json")
    reasonings_file_path = os.path.join(args.data_dir, args.libero_task_suite+"_w_mask", "cot", "filtered_reasoning_h10.json")
    with open(reasonings_file_path, "r") as f:
        reasonings = json.load(f)

    for file_path in tqdm(reasonings.keys(), desc="Merging"):
        if file_path not in bboxes:
            print(f"File path {file_path} not found in bboxes")
            continue
        if file_path not in gripper_positions:
            print(f"File path {file_path} not found in gripper_positions")
            continue
        if file_path not in primitives:
            print(f"File path {file_path} not found in primitives")
            continue
        bbox = bboxes[file_path]
        gripper_position = gripper_positions[file_path]
        primitive = primitives[file_path]

        try:
            assert len(bbox) == len(gripper_position) == len(primitive) == len(reasonings[file_path]["0"]["reasoning"]), f"Length mismatch for {file_path}: {len(bbox)}, {len(gripper_position)}, {len(primitive)}, {len(reasonings[file_path]['0']['reasoning'])}"
        except Exception as e:
            print(e)
            continue

        reasonings[file_path]["0"]["features"].update(
            {
                "bboxes": bbox,
                "gripper_position": gripper_position,
                "move_primitive": primitives
            }
        )

    target_dir = os.path.join(args.data_dir, "final_reasonings")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Saving to {target_dir}")
    target_file_path = os.path.join(target_dir, f"reasoning_{args.libero_task_suite}.json")

    with open(target_file_path, "w") as f:
        json.dump(reasonings, f)
