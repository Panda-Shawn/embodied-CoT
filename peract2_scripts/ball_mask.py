import cv2
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


task_name = "lift_ball"

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



img = np.array(image)

img_np = np.array(img)

# 转为灰度图
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

# 阈值提取高亮区域（球比较亮）
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# 可选：只在图像下半部检测（避免上方手臂干扰）
h, w = thresh.shape
mask_roi = np.zeros_like(thresh)
mask_roi[h//4:h*3//4, :] = 1
thresh = cv2.bitwise_and(thresh, thresh, mask=mask_roi)

# 查找连通区域
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找面积最大的连通区域（通常是球）
ball_mask = np.zeros_like(thresh)
if contours:
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(ball_mask, [largest], -1, 255, thickness=cv2.FILLED)

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Bright Area")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ball_mask, cmap='gray')
plt.title("Final Ball Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig("ball_mask.png")