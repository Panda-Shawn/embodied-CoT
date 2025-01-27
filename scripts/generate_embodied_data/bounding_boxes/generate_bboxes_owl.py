import argparse
import json
import os
import time
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import pipeline
from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", type=int, default=24)
parser.add_argument("--data-path", type=str)
parser.add_argument("--result-path", default="./bboxes")

args = parser.parse_args()
bbox_json_path = os.path.join(args.result_path, f"results_{args.id}_bboxes.json")

print("Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

ds = tfds.load("libero_10_no_noops", data_dir="/data/lzx/libero_new", split=f"train[{start}%:{end}%]")
print(f"data size: {len(ds)}")
print("Done.")

print("Loading Prismatic descriptions...")
results_json_path = "./descriptions/full_descriptions.json"
with open(results_json_path, "r") as f:
    results_json = json.load(f)
print("Done.")


checkpoint = "google/owlvit-base-patch16"

# detector 自动使用cuda:0
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
print("Done.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2

bbox_results_json = {}
for ep_idx, episode in enumerate(ds):

    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"ID {args.id} starting ep: {episode_id}, {file_path}")

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    episode_json = results_json[file_path][str(episode_id)]
    description = episode_json["caption"]
    seg_obj_text = episode_json["seg_obj"]
    # replace comma in seg_obj_text with dot
    seg_obj_text = [obj.strip() for obj in seg_obj_text.split(",")]

    start = time.time()
    bboxes_list = []
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())
        # print("seg_obj_text", seg_obj_text)
        results = detector(image, candidate_labels=seg_obj_text, threshold=0.01)

        bboxes = {}
        for res in results:
            lg, p, b = res["score"], res["label"], res["box"]
            b = [b["ymin"], b["xmin"], b["ymax"], b["xmax"]]
            lg = round(lg, 5)
            if p not in bboxes.keys():
                bboxes[p] = (lg, p, b)
            else:
                if lg > bboxes[p][0]:
                    bboxes[p] = (lg, p, b)
            # break
        bboxes = [(lg, p, b) for (lg, p, b) in bboxes.values()]

        bboxes_list.append(bboxes)
        break
    end = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
    # print(f"Caption: {description}")
    break
