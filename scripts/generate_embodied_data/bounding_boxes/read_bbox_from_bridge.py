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


bbox_json_path = '/home/lwh/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/snapshots/854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json'
image_label = 'image_0'
output_image_path = './bridge_bboxes'
# 加载数据集
ds = tfds.load("bridge_orig", data_dir="/data/lzx/bridge_dataset", split=f"train[{0}%:{100}%]")


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

with open(bbox_json_path, "r") as f:
    bbox_results_json = json.load(f)
    
for ep_idx, episode in enumerate(ds):
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    bboxes = bbox_results_json[file_path][str(episode_id)]['features']["bboxes"]
    print(f"episode id {bbox_results_json[file_path][str(episode_id)]['metadata']['n_steps']}")
    print(f'len {len(bboxes)}')
    print(f'real episode id {episode_id}')
    print(f'len steps {len(episode["steps"])}')
    for step_idx, step in enumerate(episode["steps"]):
        
        image = Image.fromarray(step["observation"][image_label].numpy())
        

        # 绘制边界框并保存图片
        os.makedirs(os.path.join(output_image_path, f"./{episode_id}"), exist_ok=True)
        output_path = os.path.join(output_image_path, f"./{episode_id}/ep_{episode_id}_step_{step_idx}.jpg")
        draw_boxes(image, bboxes[step_idx], output_path)
        print(f'finish ep {ep_idx} step {step_idx}')
    if ep_idx == 10:
        break
    