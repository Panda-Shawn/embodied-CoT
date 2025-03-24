import json
import h5py
from tqdm import tqdm
import argparse
import glob
import mediapy
from gripper_positions.cam_pose import CAM_POSES, IMAGE_SIZE
from gripper_positions.cam_utils import (
    calculate_2d_position,
    calculate_camera_intrinsics,
)
import numpy as np
from utils import NumpyFloatValuesEncoder, draw_gripper_position
import os
from env_utils import process_single_image
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from scipy.spatial.transform import Rotation


image_dims = (256, 256)


def label_single_task(data_path, episode, debug=False):
    print(f"Processing {data_path} ...")

    gripper_pos_results_json = {}
    gripper_pos = []
    for i, step in enumerate(episode["steps"]):
        image = step["observation"]["image"].numpy().astype(np.uint8)
        gripper_positions_3d = step["observation"]["gripper_pose"].numpy()[:3]
        camera_transform_matrix = step["camera_extrinsics"].numpy()
        camera_pos = camera_transform_matrix[:3, 3]
        camera_quat = Rotation.from_matrix(camera_transform_matrix[:3, :3]).as_quat()
        camera_intrinsics = step["camera_intrinsics"].numpy()

        # Project tcp position in the image
        u, v = calculate_2d_position(
            gripper_positions_3d,
            camera_pos,
            camera_quat,
            camera_intrinsics,
            scalar_first=False,
        )
        gripper_pos.append([int(u), int(v)])
        if debug:
            # 使用 Matplotlib 可视化
            plt.figure(figsize=(8, 6))
            plt.imshow(image)  # 显示图像
            plt.scatter(
                gripper_pos[-1][0],
                gripper_pos[-1][1],
                c="red",
                s=50,
                label="Gripper Position",
            )  # 标记点
            plt.legend()
            plt.axis("off")
            plt.savefig(
                f"vis_gpos/output_ep_{data_path.replace('/', '_')}_step_{i}.png", bbox_inches="tight", dpi=300
            )  # bbox_inches 去掉多余边距，dpi 控制分辨率
            plt.close()

    gripper_pos_results_json[data_path] = gripper_pos
    # origin_data.close()
    return gripper_pos_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ds = tfds.load(
        "colosseum_dataset",
        data_dir=args.dataset_dir,
        split=f"train[{0}%:{100}%]",
    )

    results = {}
    for episode in tqdm(ds):
        data_path = episode["episode_metadata"]["file_path"].numpy().decode()
        results_json = label_single_task(data_path, episode, args.debug)
        results.update(results_json)
        if args.debug:
            break

    if args.results_path is None:
        bbox_dir = os.path.join(args.dataset_dir, "cot")
        os.makedirs(bbox_dir, exist_ok=True)
        args.results_path = os.path.join(bbox_dir, "gripper_positions.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
