import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import json
import os


dataset_name = "libero_goal_no_noops"
ds = tfds.load(
    dataset_name, data_dir="/data2/lzixuan/libero_new", split=f"train[{0}%:{100}%]"
)
print(f"data size: {len(ds)}")
print("Done.")
gripper_positions_json_path = "./gripper_positions/gripper_positions.json"
vis_dir = "./vis_gpos"
os.makedirs(vis_dir, exist_ok=True)

with open(gripper_positions_json_path, "r") as gripper_positions_file:
    gripper_positions = json.load(gripper_positions_file)

for ep_idx, episode in enumerate(ds):

    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"starting ep: {episode_id}, {file_path}")

    gripper_pos = []
    for step_idx, step in enumerate(episode["steps"]):
        gripper_position = gripper_positions[file_path][str(episode_id)][step_idx]
        image = step["observation"]["image"].numpy()

        # 使用 Matplotlib 可视化
        plt.figure(figsize=(8, 6))
        plt.imshow(image)  # 显示图像
        plt.scatter(
            gripper_position[0],
            gripper_position[1],
            c="red",
            s=50,
            label="Gripper Position",
        )  # 标记点
        plt.legend()
        plt.axis("off")
        plt.savefig(
            os.path.join(vis_dir, f"step_{step_idx}.png"), bbox_inches="tight", dpi=300
        )  # bbox_inches 去掉多余边距，dpi 控制分辨率
        plt.close()

    break
