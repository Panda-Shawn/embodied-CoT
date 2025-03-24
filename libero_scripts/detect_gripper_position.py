import json
import h5py
from tqdm import tqdm
import argparse
import glob
import mediapy
from gripper_positions.libero_cam_pose import LIBERO_CAM_POSES, IMAGE_SIZE
from gripper_positions.cam_utils import (
    calculate_2d_position,
    calculate_camera_intrinsics,
)
import numpy as np
from utils import NumpyFloatValuesEncoder, draw_gripper_position
import os
from libero_utils import process_single_image
from matplotlib import pyplot as plt
from PIL import Image


image_dims = (256, 256)


def label_single_task(data_path, libero_task_suite, debug=False):
    origin_data_file = h5py.File(data_path, "r")
    origin_data = origin_data_file["data"]
    print(f"Processing {data_path} ...")
    if "libero_10" in libero_task_suite or "libero_90" in libero_task_suite:
        scene = data_path.split("/")[-1].split("SCENE")[0] + "SCENE"
    else:
        scene = "SCENE"
    gripper_pos_results_json = {}
    for episode in tqdm(origin_data.keys()):
        episode_data = origin_data[episode]
        episode_id = episode_data["episode_id"][()].decode("utf-8")
        gripper_pos = []
        image_with_gripper_pos = []
        step_nums = episode_data["actions"][()].shape[0]
        for i in range(step_nums):
            image = np.array(
                process_single_image(
                    episode_data["obs"]["agentview_rgb"][i], image_dims
                )
            ).astype(np.uint8)
            gripper_positions_3d = episode_data["obs"]["ee_pos"][i]
            camera_pos = LIBERO_CAM_POSES[libero_task_suite][scene]["agentview"]["pos"]
            camera_quat = LIBERO_CAM_POSES[libero_task_suite][scene]["agentview"][
                "quat"
            ]
            fovy = LIBERO_CAM_POSES[libero_task_suite][scene]["agentview"]["fovy"]
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

            # image = Image.fromarray(image, mode="RGB")
            # image = image.resize((224, 224))
            # scale = 224 / 256
            # u, v = u * scale, v * scale

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
                    f"vis_gpos/output_{episode_id}_step_{i}.png", bbox_inches="tight", dpi=300
                )  # bbox_inches 去掉多余边距，dpi 控制分辨率
                plt.close()

        gripper_pos_results_json[episode_id] = gripper_pos
    # origin_data.close()
    return gripper_pos_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=[
            "libero_spatial_no_noops",
            "libero_object_no_noops",
            "libero_goal_no_noops",
            "libero_10_no_noops",
            "libero_90_no_noops",
        ],
        help="LIBERO task suite. Example: libero_spatial",
    )
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_files = glob.glob(os.path.join(args.libero_dataset_dir, "*.hdf5"))
    results = {}
    for i in range(len(data_files)):
        data_path = os.path.join(args.libero_dataset_dir, data_files[i])
        results_json = label_single_task(data_path, args.libero_task_suite, args.debug)
        results.update(results_json)
        if args.debug:
            break

    if args.results_path is None:
        bbox_dir = os.path.join(args.libero_dataset_dir, "cot")
        os.makedirs(bbox_dir, exist_ok=True)
        args.results_path = os.path.join(bbox_dir, "gripper_positions.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
