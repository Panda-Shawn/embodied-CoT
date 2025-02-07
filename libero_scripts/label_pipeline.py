import subprocess
import json
import argparse
from pathlib import Path
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str)
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--action_horizon", type=int)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--vlm_model_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--api_provider", type=str)
    parser.add_argument("--enable_scene_desc", action="store_true")
    parser.add_argument("--enable_primitives", action="store_true")
    parser.add_argument("--enable_bboxes", action="store_true")
    parser.add_argument("--enable_gripper_positions", action="store_true")
    parser.add_argument("--enable_plan_subtasks", action="store_true")
    parser.add_argument("--enable_merge", action="store_true")

    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = os.path.join(args.libero_dataset_dir, "cot")
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Create results_dir: {args.results_dir}")

    # Get the dir of scripts for libero dataset labelling
    libero_scripts_dir = str(Path(__file__).parent)

    # Describe scene with VLMs
    if args.enable_scene_desc:
        desc_script_path = os.path.join(libero_scripts_dir, "describe_scene.py")
        desc_args = [
            "--results_path",
            f"{args.results_dir}/{args.libero_task_suite}_scene_desc.json",
            "--libero_dataset_dir",
            args.libero_dataset_dir,
            "--vlm_model_path",
            args.vlm_model_path,
        ]
        desc_command = ["python", desc_script_path] + desc_args
        subprocess.run(desc_command, check=True)

    # Extract motion primitives
    if args.enable_primitives:
        extract_primitives_script_path = os.path.join(
            libero_scripts_dir, "extract_primitives.py"
        )
        primitive_args = [
            "--libero_dataset_dir",
            args.libero_dataset_dir,
            "--results_path",
            f"{args.results_dir}/{args.libero_task_suite}_primitives.json",
            "--action_horizon",
            str(args.action_horizon),
        ]
        primitive_command = ["python", extract_primitives_script_path] + primitive_args
        subprocess.run(primitive_command, check=True)

    # Detect Boundary Box for all object in the scene
    if args.enable_bboxes:
        bboxes_script_path = os.path.join(libero_scripts_dir, "generate_bboxes.py")
        bboxes_args = [
            "--libero_dataset_dir",
            args.libero_dataset_dir,
            "--results_path",
            f"{args.results_dir}/{args.libero_task_suite}_bboxes.json",
            # "--caption_path",
            # f"{args.results_dir}/{args.libero_task_suite}_scene_desc.json",
            "--debug",
        ]
        bboxes_command = ["python", bboxes_script_path] + bboxes_args
        subprocess.run(bboxes_command, check=True)

    # Detect the gripper pixel position in the image
    if args.enable_gripper_positions:
        gripper_pos_script_path = os.path.join(
            libero_scripts_dir, "detect_gripper_position.py"
        )
        gripper_args = [
            "--libero_task_suite",
            args.libero_task_suite,
            "--libero_dataset_dir",
            args.libero_dataset_dir,
            "--results_path",
            f"{args.results_dir}/{args.libero_task_suite}_gripper_pos.json",
            # "--debug",
        ]
        gripper_command = ["python", gripper_pos_script_path] + gripper_args
        subprocess.run(gripper_command, check=True)

    # Label subtasks with API of LLMs
    if args.enable_plan_subtasks:
        label_pipeline_script_path = os.path.join(
            libero_scripts_dir, "batch_generate_plan_subtasks.py"
        )
        label_args = [
            "--batch_size",
            str(args.batch_size),
            "--action_horizon",
            str(args.action_horizon),
            "--force_regenerate",
            "--api_provider",
            args.api_provider,
            "--libero_dataset_dir",
            args.libero_dataset_dir,
            "--libero_scene_desc_path",
            f"{args.results_dir}/{args.libero_task_suite}_scene_desc.json",
            "--libero_primitives_path",
            f"{args.results_dir}/{args.libero_task_suite}_primitives.json",
            "--llms_response_save_dir",
            f"{args.results_dir}/llms_response",
            "--results_path",
            f"{args.results_dir}/{args.libero_task_suite}_plan_subtasks.json",
        ]
        label_command = ["python", label_pipeline_script_path] + label_args
        subprocess.run(label_command, check=True)

    # Merge all Chain-of-Thought into a single file
    if args.enable_merge:
        merge_script_path = os.path.join(libero_scripts_dir, "merge_plan_subtasks.py")
        merge_cot = {}
        with open(
            f"{args.results_dir}/{args.libero_task_suite}_scene_desc.json", "r"
        ) as f:
            scene_desc = json.load(f)
        with open(
            f"{args.results_dir}/{args.libero_task_suite}_primitives.json", "r"
        ) as f:
            primitives = json.load(f)
        with open(
            f"{args.results_dir}/{args.libero_task_suite}_plan_subtasks.json", "r"
        ) as f:
            plan_subtasks = json.load(f)
        print(scene_desc.keys())
        print("-" * 100)
        print(primitives.keys())
        print("-" * 100)
        print(plan_subtasks.keys())
        assert scene_desc.keys() == primitives.keys()
        for key in scene_desc.keys():
            merge_cot[key] = {
                "task_description": scene_desc[key]["task_description"],
                "scene_desc": scene_desc[key]["caption"],
                "primitives": primitives[key],
                "plan_subtasks": plan_subtasks[key],
            }
        with open(f"{args.results_dir}/{args.libero_task_suite}_cot.json", "w") as f:
            json.dump(merge_cot, f)
