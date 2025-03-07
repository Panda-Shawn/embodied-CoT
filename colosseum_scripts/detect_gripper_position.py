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
import pickle


image_dims = (256, 256)


def label_single_task(data_path, debug=False, furniture=None):
    origin_data_file = open(data_path, "rb")
    origin_data = pickle.load(origin_data_file)
    if furniture is None:
        furniture = data_path.split("/")[-3]
    # time = data_path.split("/")[-2]
    print(f"Processing {data_path} ...")

    gripper_pos_results_json = {}
    gripper_pos = []
    step_nums = len(origin_data["actions"])
    for i in range(step_nums):
        image = np.array(
            origin_data["observations"][i]["color_image2"],
        ).astype(np.uint8)
        gripper_positions_3d = origin_data["observations"][i]["robot_state"]["ee_pos"]
        camera_pos = CAM_POSES[furniture]["front"]["pos"]
        camera_quat = CAM_POSES[furniture]["front"][
            "quat"
        ]
        fovy = CAM_POSES[furniture]["front"]["fovy"]
        resolution = IMAGE_SIZE

        camera_intrinsics = calculate_camera_intrinsics(fovy, resolution)

        # Project tcp position in the image
        u, v = calculate_2d_position(
            gripper_positions_3d,
            camera_pos,
            camera_quat,
            camera_intrinsics,
            scalar_first=True,
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
    parser.add_argument("--furniture", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_files = glob.glob(os.path.join(args.dataset_dir, "*/*.pkl"))
    results = {}
    for i in tqdm(range(len(data_files))):
        data_path = os.path.join(args.dataset_dir, data_files[i])
        results_json = label_single_task(data_path, args.debug, args.furniture)
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
