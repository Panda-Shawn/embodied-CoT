import cv2
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


task_name = "handover_item"

ds = tfds.load(
    task_name,
    data_dir=f"/data/lzx/peract2/{task_name}_dir",
    split=f"train[{0}%:{100}%]",
)

for i, episode in enumerate(ds):
    if i == 1:
        print("file path: ", episode["episode_metadata"]["file_path"].numpy().decode())
        for j, step in enumerate(episode["steps"]):
            lang_instruction = step["language_instruction"].numpy().decode()
            print(lang_instruction)
            image = Image.fromarray(step["observation"]["front_image"].numpy())
            break
        break


# Step 1: 加载图像并转换为 RGB 数组
img = image
img_np = np.array(img)

# Step 2: 设定目标颜色和容差
target_rgb = np.array([122, 7, 122])  # #7A077A
tolerance = 80  # 容差范围，可调节，越大越宽松

# Step 3: 计算每个像素与目标颜色的欧氏距离
distance = np.linalg.norm(img_np - target_rgb, axis=2)

# Step 4: 生成 mask：在容差范围内设为1，其余为0
binary_mask = (distance < tolerance).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(binary_mask, cmap='gray')  # mask 是 0 和 1 的二值图
plt.title("Purple Region Mask")
plt.axis('off')
plt.savefig('color_mask.png')
