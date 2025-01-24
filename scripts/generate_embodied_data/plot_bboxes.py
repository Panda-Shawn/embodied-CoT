import json
import os
import re
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode
from scripts.generate_embodied_data.gripper_positions import get_corrected_positions_episode
import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt


def show_box(box, ax, meta, color):
    x0, y0 = box["xmin"], box["ymin"]
    w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, "{:.3f}".format(meta["score"]), color="white")


def plot_bboxes(builder, episode_ids, save_path="reasonings.json"):

    with open("bounding_boxes/bboxes/full_bboxes.json", "r") as bboxes_file:
        bboxes = json.load(bboxes_file)


    for i in episode_ids:
        entry = build_single_reasoning(i, builder, lm, captions_dict, bboxes, gripper_positions)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        print("computed reasoning:", entry)

    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f)


if __name__ == "__main__":
    builder = tfds.builder(name="libero_10_no_noops", data_dir="/data/lzx/libero_new")
    generate_reasonings(builder, list(range(10)))
