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
)

from libero_utils import process_single_image

warnings.filterwarnings("ignore")

image_dims = (256, 256)


def label_single_task(data_path, debug=False):
    origin_data_file = h5py.File(data_path, "r")
    origin_data = origin_data_file["data"]
    print(f"Processing {data_path} ...")

    bbox_results_json = {}
    for episode in tqdm(origin_data.keys()):
        episode_data = origin_data[episode]
        episode_id = episode_data["episode_id"][()].decode("utf-8")
        instance_names = episode_data["instance_names"][()]
        instance_names = decode_instance_names(instance_names)
        instance_id_to_names = {}
        for i, instance in enumerate(instance_names):
            instance_id_to_names[i + 1] = instance
        step_nums = episode_data["actions"][()].shape[0]
        bboxes_list = []
        for i in range(step_nums):
            image = Image.fromarray(
                np.array(
                    process_single_image(
                        episode_data["obs"]["agentview_rgb"][i], image_dims
                    )
                ),
                mode="RGB",
            )
            mask = Image.fromarray(
                np.array(episode_data["obs"]["agentview_segmentation_instance"][i])
                .squeeze()
                .astype(np.uint8),
                mode="L",
            )
            mask = mask.resize(image_dims)
            mask = mask.rotate(180)

            if debug:
                img_dir = f"vis_bboxes/img_episode_{episode_id}"
                os.makedirs(img_dir, exist_ok=True)
                image.save(os.path.join(img_dir, f"raw_image_step_{i}.png"))

            bboxes = mask_to_bboxes(np.array(mask), instance_id_to_names)
            if debug:
                bbox_dir = f"vis_bboxes/bbox_episode_{episode_id}"
                draw_bounding_boxes(image, bboxes, os.path.join(bbox_dir, f"bboxes_{episode_id}_{i}.png"))
            bboxes_list.append(bboxes)
        bbox_results_json[episode_id] = bboxes_list
    origin_data_file.close()
    return bbox_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_files = glob.glob(os.path.join(args.libero_dataset_dir, "*.hdf5"))
    results = {}
    for i in range(len(data_files)):
        data_path = os.path.join(args.libero_dataset_dir, data_files[i])
        results_json = label_single_task(data_path, args.debug)
        results.update(results_json)

    if args.results_path is None:
        bbox_dir = os.path.join(args.libero_dataset_dir, "cot")
        os.makedirs(bbox_dir, exist_ok=True)
        args.results_path = os.path.join(bbox_dir, "bboxes.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
