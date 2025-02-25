import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--results-path", default="./descriptions")

args = parser.parse_args()

device = f"cuda:{args.gpu}"
warnings.filterwarnings("ignore")

# Load Qwen2.5-VL model and processor
print("Loading Qwen2.5-VL model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# Load Bridge V2 dataset
ds = tfds.load(
    "libero_10_no_noops",
    data_dir="/data/lzx/libero_new",
    split=f"train[{start}%:{end}%]",
)

results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")

def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

def create_seg_prompt(caption):
    user_prompt = f"The caption is: '{caption}' List only the objects found in the caption separated by dots."
    return user_prompt

def generate_with_qwen_vl(image, prompt_text):
    """
    Generate a response using the Qwen2.5-VL model.

    Args:
        image (PIL.Image): The input image.
        prompt_text (str): The prompt text for generation.

    Returns:
        The generated text.
    """
    # Prepare messages for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

results_json = {}
for i, episode in tqdm(enumerate(ds)):
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())

        # Generate caption
        user_prompt = create_user_prompt(lang_instruction)
        caption = generate_with_qwen_vl(image, user_prompt)

        # Generate segmentation objects
        seg_prompt = create_seg_prompt(caption)
        seg_obj = generate_with_qwen_vl(image, seg_prompt)

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