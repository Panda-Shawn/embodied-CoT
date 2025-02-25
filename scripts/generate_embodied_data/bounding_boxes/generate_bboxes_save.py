import argparse
import json
import os
import time
import warnings
from PIL import Image, ImageDraw, ImageFont
import tensorflow_datasets as tfds
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", type=int, default=24)
parser.add_argument("--data-path", type=str)
parser.add_argument("--result-path", default="./bboxes")
parser.add_argument("--output-image-path", default="./annotated_images")  # 新增：保存标注图片的路径

image_label = 'image_0'

args = parser.parse_args()
bbox_json_path = os.path.join(args.result_path, f"results_{args.id}_bboxes.json")

# 创建保存标注图片的目录
os.makedirs(args.output_image_path, exist_ok=True)

print("Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# 加载数据集
ds = tfds.load("bridge_orig", data_dir="/data/lzx/bridge_dataset", split=f"train[{start}%:{end}%]")
print(f"data size: {len(ds)}")
print("Done.")

print("Loading Prismatic descriptions...")
# results_json_path = "./descriptions/results_0.json"
results_json_path = "/home/lwh/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/snapshots/854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
with open(results_json_path, "r") as f:
    results_json = json.load(f)
print("Done.")

print(f"Loading gDINO to device {args.gpu}...")
model_id = "IDEA-Research/grounding-dino-base"
device = f"cuda:{args.gpu}"

processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2

# 定义绘制边界框的函数
def draw_boxes(image, bboxes, output_path):
    """
    在图片上绘制边界框并保存
    :param image: PIL 图像对象
    :param bboxes: 边界框列表，格式为 [(logit, phrase, box)]
    :param output_path: 保存图片的路径
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # 使用默认字体

    for lg, phrase, box in bboxes:
        # 绘制边界框
        draw.rectangle(box, outline="red", width=2)
        # 在边界框上方绘制标签和置信度
        label = f"{phrase} ({lg:.2f})"
        draw.text((box[0], box[1] - 10), label, fill="red", font=font)

    # 保存图片
    image.save(output_path)

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
    seg_obj_text = seg_obj_text.replace(",", ".")

    start = time.time()
    bboxes_list = []
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"][image_label].numpy())
        inputs = processor(
            images=image,
            text=seg_obj_text,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = {}
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = round(lg, 5)
            if p not in bboxes.keys():
                bboxes[p] = (lg, p, b)
            else:
                if lg > bboxes[p][0]:
                    bboxes[p] = (lg, p, b)
        bboxes = [(lg, p, b) for (lg, p, b) in bboxes.values()]

        bboxes_list.append(bboxes)

        # 绘制边界框并保存图片
        os.makedirs(os.path.join(args.output_image_path, f"./{episode_id}"), exist_ok=True)
        output_image_path = os.path.join(args.output_image_path, f"./{episode_id}/ep_{episode_id}_step_{step_idx}.jpg")
        draw_boxes(image, bboxes, output_image_path)

    end = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
    