import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--libero_cot_path", type=str, default=None)
    parser.add_argument("--libero_gripper_path", type=str, default=None)
    parser.add_argument("--libero_bbox_path", type=str, default=None)
    parser.add_argument("--add_gripper_position", action="store_true")
    parser.add_argument("--add_bounding_box", action="store_true")
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    cot_dir = os.path.join(args.libero_dataset_dir, "cot")
    if args.libero_cot_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_cot_path = os.path.join(
            cot_dir, f"chain_of_thought_h{args.action_horizon}.json"
        )

    with open(args.libero_cot_path, "r") as f:
        cot_data = json.load(f)

    suffix = ""
    if args.add_gripper_position:
        if args.libero_gripper_path is None:
            os.makedirs(cot_dir, exist_ok=True)
            args.libero_gripper_path = os.path.join(cot_dir, "gripper_positions.json")

        with open(args.libero_gripper_path, "r") as f:
            gripper_data = json.load(f)

        for k in cot_data.keys():
            cot_data[k]["0"]["features"]["gripper_position"] = gripper_data[k]

        suffix += "_gripper"

    if args.add_bounding_box:
        if args.libero_bbox_path is None:
            os.makedirs(cot_dir, exist_ok=True)
            args.libero_bbox_path = os.path.join(cot_dir, "bboxes.json")
        with open(args.libero_bbox_path, "r") as f:
            bbox_data = json.load(f)
        for k in cot_data.keys():
            cot_data[k]["0"]["features"]["bbox"] = bbox_data[k]
        suffix += "_bbox"

    if args.result_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.result_path = os.path.join(
            cot_dir, f"chain_of_thought_h{args.action_horizon}{suffix}.json"
        )

    with open(args.result_path, "w") as f:
        json.dump(cot_data, f)
