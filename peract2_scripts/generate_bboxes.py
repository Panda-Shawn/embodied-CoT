import argparse
import json
import os
import warnings
import h5py
import glob
from tqdm import tqdm
import numpy as np

from PIL import Image
from utils import (
    NumpyFloatValuesEncoder,
    draw_bounding_boxes,
    decode_instance_names,
    mask_to_bboxes,
    show_box,
    mask_or_color_to_bboxes
)

from env_utils import process_single_image, INSTANCE_ID_TO_NAMES_AND_COLORS

import pickle
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

warnings.filterwarnings("ignore")

image_dims = (224, 224)


def label_single_task(data_path, episode, debug=False):
    print(f"Processing {data_path} ...")

    bbox_results_json = {}
    task = data_path.split("_val/")[0].split("/")[-1]
    instance_id_to_names = INSTANCE_ID_TO_NAMES_AND_COLORS[task]
    instance_names = instance_id_to_names.values()
    bboxes_list = []
    for i, step in enumerate(episode["steps"]):
        image = Image.fromarray(
            step["observation"]["front_image"].numpy().astype(np.uint8),
            mode="RGB",
        )
        mask = step["observation"]["front_mask"].numpy().astype(np.uint8)

        # bboxes = mask_to_bboxes(mask, instance_id_to_names)
        bboxes = mask_or_color_to_bboxes(np.array(image), mask, instance_id_to_names)
        # import pdb; pdb.set_trace()
        if debug:
            # Plot the image using matplotlib
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

            # Draw bounding boxes
            for text, bbox in bboxes:
                show_box(bbox, ax, text, "red")
            
            output_path = f"vis_bboxes/output_ep_{data_path.replace('/', '_')}_step_{i}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        bboxes_list.append(bboxes)
    bbox_results_json[data_path] = bboxes_list
    return bbox_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ds = tfds.load(
        args.dataset_dir.split("/")[-1].split("_dir")[0],
        data_dir=args.dataset_dir,
        split=f"train[{0}%:{100}%]",
    )

    results = {}
    for episode in tqdm(ds):
        data_path = episode["episode_metadata"]["file_path"].numpy().decode()
        results_json = label_single_task(data_path, episode, args.debug)
        results.update(results_json)
        if args.debug:
            break

    if args.results_path is None:
        bbox_dir = os.path.join(args.dataset_dir, "cot")
        os.makedirs(bbox_dir, exist_ok=True)
        args.results_path = os.path.join(bbox_dir, "bboxes.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
