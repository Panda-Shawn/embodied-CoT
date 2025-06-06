import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from prismatic import load

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--results-path", default="./descriptions_object")

args = parser.parse_args()

device = f"cuda:{args.gpu}"
hf_token = ""
vlm_model_id = "prism-dinosiglip+7b"

warnings.filterwarnings("ignore")


split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# Load Bridge V2
ds = tfds.load(
    "libero_object_no_noops",
    data_dir="/data/lzx/libero_new",
    split=f"train[{start}%:{end}%]",
)

# Load Prismatic VLM
print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)

results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")


def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the objects in this scene."]
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    
    return user_prompt


def create_seg_prompt(caption):
    user_prompt = f"The caption is: '{caption}' List only the objects found in the caption separated by dots."
    return user_prompt


results_json = {}
# add enumerate to get episode_id
# for episode in tqdm(ds):
for i, episode in tqdm(enumerate(ds)):
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    # episode_id = i
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())

        # user_prompt = "Describe the objects in this scene. Be specific."
        user_prompt = create_user_prompt(lang_instruction)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        caption = vlm.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=64,
            min_length=1,
        )

        user_prompt = create_seg_prompt(caption)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()
        seg_obj = vlm.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=64,
            min_length=1,
        )
        break

    episode_json = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "caption": caption,
        "seg_obj": seg_obj
    }

    if file_path not in results_json.keys():
        results_json[file_path] = {}

    results_json[file_path][int(episode_id)] = episode_json

    with open(results_json_path, "w") as f:
        json.dump(results_json, f, cls=NumpyFloatValuesEncoder)
