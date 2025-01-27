import tensorflow_datasets as tfds
import json
from scripts.generate_embodied_data.bounding_boxes.utils import NumpyFloatValuesEncoder
import time

from scripts.generate_embodied_data.gripper_postions.libero_cam_pose import LIBERO_CAM_POSES, IMAGE_SIZE
from scripts.generate_embodied_data.gripper_postions.cam_utils import calculate_2d_position, calculate_camera_intrinsics


image_label = "image"

if __name__=="__main__":
    dataset_name = "libero_spatial_no_noops" # "libero_10_no_noops"
    ds = tfds.load(dataset_name, data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")
    gripper_positions_json_path = "./gripper_positions/gripper_positions.json"

    start = time.time()
    gripper_positions_json = {}
    for ep_idx, episode in enumerate(ds):

        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")

        gripper_pos = []
        for step in episode["steps"]:
            state = step["observation"]["state"].numpy()
            gripper_positions_3d = state[:3]
            if "SCENE" in file_path:
                scene_name = file_path.split("/")[-1].split("SCENE")[0] + "SCENE"
            elif "libero_spatial" in file_path:
                scene_name = "SCENE"
            camera_pos = LIBERO_CAM_POSES[dataset_name][scene_name]["agentview"]["pos"]
            camera_quat = LIBERO_CAM_POSES[dataset_name][scene_name]["agentview"]["quat"]

            fovy = LIBERO_CAM_POSES[dataset_name][scene_name]["agentview"]["fovy"]
            resolution = IMAGE_SIZE

            camera_intrinsics = calculate_camera_intrinsics(fovy, resolution)

            # 计算 2D 坐标
            u, v = calculate_2d_position(gripper_positions_3d, camera_pos, camera_quat, camera_intrinsics, scalar_first=True)
            gripper_pos.append([int(u), int(v)])

        if file_path not in gripper_positions_json.keys():
            gripper_positions_json[file_path] = {}

        gripper_positions_json[file_path][int(episode_id)] = gripper_pos
        end = time.time()
        with open(gripper_positions_json_path, "w") as f:
            json.dump(gripper_positions_json, f, cls=NumpyFloatValuesEncoder)
        print(f"finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
        