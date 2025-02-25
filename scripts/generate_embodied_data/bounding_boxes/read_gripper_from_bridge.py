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
import time

import os

json_path = '/data/lwh/gripper_positions/gripper_positions_furniture_bench_dataset_converted_externally_to_rlds.json'
ds = tfds.load("furniture_bench_dataset_converted_externally_to_rlds", data_dir="/data/lzx/oxe_mods/", split=f"train[{0}%:{100}%]")

with open(json_path, 'r') as f:
    data = json.load(f)

image_label = 'image'
TIM = 0
for ep_idx, episode in enumerate(ds):
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    images = [step["observation"][image_label] for step in episode["steps"]]
    images = [img.numpy() for img in images]
    
    # pos  = data[file_path][str(episode_id)]['features']["gripper_position"]
    pos  = data[file_path][str(episode_id)]
    
    
    output_file = f"./gripper_furniture/output_video_{TIM}.mp4"  # 输出视频文件名
    TIM += 1
    fps = 10  # 帧率
    frame_size = (images[0].shape[1], images[0].shape[0])  # 视频帧的宽度和高度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # 在图像上绘制点并写入视频
    for img, p in zip(images, pos):
        img_with_circle = cv2.circle(img, (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
        out.write(img_with_circle)  # 将帧写入视频
    print(f'success')
    # 释放 VideoWriter 对象
    out.release()   
    
    if ep_idx == 8:
        break

    
    