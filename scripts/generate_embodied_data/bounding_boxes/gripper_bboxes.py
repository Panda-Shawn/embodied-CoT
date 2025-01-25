import cv2
import matplotlib
import mediapy
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline
import tensorflow_datasets as tfds
import json
from utils import NumpyFloatValuesEncoder
import time

import os



checkpoint = "google/owlvit-base-patch16"

# detector 自动使用cuda:0
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

device = detector.device
sam_model.to(device)

# print("Detector device:", detector.device)
# print("SAM model device:", sam_model.device)

image_dims = (256, 256)
image_label = "image"


def get_bounding_boxes(img, prompt="the silver robotic gripper"):

    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)
  
    return predictions


def process_trajectory(episode):
    step = next(iter(episode["steps"]))
    img = step["observation"][image_label]
    img = Image.fromarray(img.numpy())
    
    predictions = get_bounding_boxes(img)

    return predictions[0]


if __name__=="__main__":
    ds = tfds.load("libero_10_no_noops", data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")
    gripper_positions_json_path = "./gripper_positions/gripper_bboxes.json"

    start = time.time()
    gripper_positions_json = {}
    for ep_idx, episode in enumerate(ds):

        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")

        bbox = process_trajectory(episode)

        if file_path not in gripper_positions_json.keys():
            gripper_positions_json[file_path] = {}

        gripper_positions_json[file_path][int(episode_id)] = bbox
        end = time.time()
        # print(gripper_positions_json)
        with open(gripper_positions_json_path, "w") as f:
            json.dump(gripper_positions_json, f, cls=NumpyFloatValuesEncoder)
        print(f"finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
        