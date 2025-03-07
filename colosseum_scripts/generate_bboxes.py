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
    show_box
)

from env_utils import process_single_image, INSTANCE_ID_TO_NAMES

import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

image_dims = (256, 256)


def label_single_task(data_path, debug=False, furniture=None):
    origin_data_file = open(data_path, "rb")
    origin_data = pickle.load(origin_data_file)
    if furniture is None:
        furniture = data_path.split("/")[-3]
    print(f"Processing {data_path} ...")

    bbox_results_json = {}

    instance_id_to_names = INSTANCE_ID_TO_NAMES[furniture]
    instance_names = instance_id_to_names.values()
    instance_names = decode_instance_names(instance_names)
    step_nums = len(origin_data["actions"])
    bboxes_list = []
    for i in range(step_nums):
        image = Image.fromarray(
            origin_data["observations"][i]["color_image2"],
            mode="RGB",
        )
        mask = origin_data["observations"][i]["seg_image2"]

        bboxes = mask_to_bboxes(mask, instance_id_to_names)
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
    origin_data_file.close()
    return bbox_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_files = glob.glob(os.path.join(args.dataset_dir, "*/*.pkl"))
    results = {}
    for i in tqdm(range(len(data_files))):
        data_path = os.path.join(args.dataset_dir, data_files[i])
        results_json = label_single_task(data_path, args.debug, args.furniture)
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
