import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import (
    NumpyFloatValuesEncoder,
    decode_instance_names,
)
import glob

from prismatic import load

from libero_utils import process_single_image

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
try:
    hf_token = Path("./.hf_token").read_text().strip()
except FileNotFoundError:
    hf_token = None

image_dims = (256, 256)


def create_user_prompt(lang_instruction, instance_names):
    user_prompt = (
        f"The robot task is: {lang_instruction}. "
        # "Analyze the image of a scene where a robot arm is about to perform this task on a wooden floor or table. "
        "Analyze the image of a scene on a wooden floor or table. "
        f"There are {', '.join(instance for instance in instance_names)} in the scene. "
        "Briefly describe the objects on the wooden floor or table and their spatial relations to each other. "
    )
    return user_prompt
    # user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # # user_prompt = "Briefly describe the objects in this scene."]
    # lang_instruction = lang_instruction.strip()
    # if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
    #     lang_instruction = lang_instruction[:-1]
    # if len(lang_instruction) > 0 and " " in lang_instruction:
    #     user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    # return user_prompt


def label_single_task(vlm, data_path):
    origin_data_file = h5py.File(data_path, "r")
    origin_data = origin_data_file["data"]

    results_json = {}

    for episode in tqdm(origin_data.keys()):
        episode_data = origin_data[episode]
        episode_id = episode_data["episode_id"][()].decode("utf-8")
        task_description = episode_data["task_description"][()].decode("utf-8")
        image = Image.fromarray(
            np.array(
                process_single_image(
                    episode_data["obs"]["agentview_rgb"][0], image_dims
                )
            ),
            mode="RGB",
        )
        instance_names = episode_data["instance_names"][()]
        instances = decode_instance_names(instance_names)
        user_prompt = create_user_prompt(task_description, instances)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        caption = vlm.generate(
            image,
            prompt_text,
            do_sample=False,
            max_new_tokens=128,
            min_length=1,
        )
        # print(caption)

        results_json[episode_id] = {
            "caption": caption,
            "task_description": task_description,
        }
        # break

    origin_data_file.close()

    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--vlm_model_path", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # Load Prismatic VLM
    device = "cuda"
    print(f"Loading Prismatic VLM ({args.vlm_model_path})...")
    # vlm = load(Path(args.vlm_model_path), hf_token=hf_token)
    vlm = load(args.vlm_model_path, hf_token=hf_token) # prism-dinosiglip+7b
    vlm = vlm.to(device, dtype=torch.bfloat16)

    # Label data files one by one
    data_files = glob.glob(os.path.join(args.libero_dataset_dir, "*.hdf5"))
    results = {}
    for i in range(len(data_files)):
        data_path = os.path.join(args.libero_dataset_dir, data_files[i])
        results_json = label_single_task(vlm, data_path)
        results.update(results_json)

    if args.results_path is None:
        cot_dir = os.path.join(args.libero_dataset_dir, "cot")
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(cot_dir, "scene_descriptions.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)

    print("Finished scene descriptions.")
