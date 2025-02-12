import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import json
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data/lzx/tensorflow_datasets")
    parser.add_argument("--reasoning_dir", type=str, default="/data/lzx/embodied-CoT/scripts/generate_embodied_data/new_reasonings/final_reasonings")
    args = parser.parse_args()

    reasoning_file_path = os.path.join(args.reasoning_dir, f"reasoning_{args.libero_task_suite}.json")
    with open(reasoning_file_path, "r") as f:
        reasonings = json.load(f)

    ds = tfds.load(
        args.libero_task_suite, data_dir=args.data_dir, split=f"train[{0}%:{100}%]"
    )
    print(f"data size: {len(ds)}")
    print("Done.")



    for ep_idx, episode in enumerate(ds):

        episode_id = episode["episode_metadata"]["episode_id"].numpy().decode()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")

        gripper_pos = []
        for step_idx, step in enumerate(episode["steps"]):
            gripper_position = reasonings[file_path][str(episode_id)]["features"]["gripper_position"][step_idx]
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
                os.path.join(f"vis_gpos/step_{step_idx}.png"), bbox_inches="tight", dpi=300
            )  # bbox_inches 去掉多余边距，dpi 控制分辨率
            plt.close()

        break
