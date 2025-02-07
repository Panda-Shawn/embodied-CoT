# This is demo for using mask of gripper from LIBERO dataset.

import h5py
from PIL import Image
import numpy as np
from utils import split_gripper_mask
import re


data_path = "/data/lx/libero_dataset/openvla_modified/libero_object_mask_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
origin_data_file = h5py.File(data_path, "r")
origin_data = origin_data_file["data"]

for episode in origin_data.keys():
    episode_data = origin_data[episode]
    instance_names = episode_data["instance_names"][()]
    instance_id_to_names = {}
    for i, instance in enumerate(instance_names):
        name = instance.decode("utf-8")
        name = re.sub(r"(_\d+)+$", "", name)
        instance_id_to_names[i + 1] = name
    image = Image.fromarray(
        np.array(episode_data["obs"]["agentview_rgb"][0]), mode="RGB"
    )
    mask = np.array(episode_data["obs"]["agentview_segmentation_instance"][0]).squeeze()
    mask = split_gripper_mask(mask, instance_id_to_names)
    mask = (mask.astype(np.uint16) * 255).clip(0, 255).astype(np.uint8)
    mask = Image.fromarray(mask, mode="L")
    image = image.rotate(180)
    mask = mask.rotate(180)
    image.save("image.png")
    mask.save("mask.png")
    break
