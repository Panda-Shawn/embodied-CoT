import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os

# 加载数据集
ds = tfds.load("bridge_orig", data_dir="/data/lzx/bridge_dataset", split=f"train[{0}%:{100}%]")

# 定义图像标签
image_labels = ["image_1", "image_2", "image_3"]

# 保存路径
save_dir = "./badimages"
os.makedirs(save_dir, exist_ok=True)

# 遍历数据集中的前 10 条轨迹
for episode_idx, episode in enumerate(ds):
    if episode_idx >= 10:  # 只保存前 10 条轨迹
        break

    # 为每条轨迹创建一个文件夹
    episode_dir = os.path.join(save_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)

    # 提取图像
    for step_idx, step in enumerate(episode["steps"]):
        for image_label in image_labels:
            # 获取图像并转换为 NumPy 数组
            image = step["observation"][image_label].numpy()

            # 将 NumPy 数组保存为图片
            image_path = os.path.join(episode_dir, f"step_{step_idx}_{image_label}.png")
            Image.fromarray(image).save(image_path)

    print(f"Saved images for episode {episode_idx}")