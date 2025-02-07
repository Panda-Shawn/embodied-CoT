import argparse
import numpy as np
import h5py
import os
import json
import glob
from tqdm import tqdm


def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "rotate counterclockwise", 0: None, 1: "rotate clockwise"},
        {},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {-1: "open gripper", 0: None, 1: "close gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move, threshold=0.03):
    diff = (move[-1] - move[0]) / len(move) * 10

    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    diff[3:6] /= 6

    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


def get_move_primitives_episode(episode):
    steps = list(episode["steps"])
    states = np.array([step["observation"]["state"] for step in steps])
    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move) for move in move_trajs]
    primitives.append(primitives[-1])
    return primitives


def extract_single_task(data_path, action_horizon, debug: bool = False):
    origin_data_file = h5py.File(data_path, "r")
    origin_data = origin_data_file["data"]
    results_json = {}

    for episode in origin_data.keys():
        episode_data = origin_data[episode]
        episode_ee_states = episode_data["obs"]["ee_states"][()]
        episode_action = episode_data["actions"][()]
        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        gripper_action = episode_action[:, -1:]
        episode_id = episode_data["episode_id"][()].decode("utf-8")

        episode_states = np.concatenate([episode_ee_states, gripper_action], axis=-1)

        move_trajs = [
            episode_states[i : i + action_horizon]
            for i in range(len(episode_states) - 1)
        ]
        primitives_list = [classify_movement(move)[0] for move in move_trajs]
        primitives_list.append(primitives_list[-1])

        results_json[episode_id] = primitives_list
        if debug:
            # (ep_len, 256, 256, 3)
            rgb_frames = episode_data["obs"]["agentview_rgb"][()]
            rgb_frames = np.rot90(rgb_frames, k=2, axes=(1, 2))  # Rotate 180 degrees

            # Add text overlay to each frame
            import cv2

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # White
            thickness = 1

            video_dir = "debug_videos/primitives"
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{episode_id}.mp4")
            video_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 256)
            )

            for frame, primitive in zip(rgb_frames, primitives_list):
                frame = frame.copy()  # Make a copy to avoid modifying original
                # Get text size to position at bottom
                (text_width, text_height), _ = cv2.getTextSize(
                    primitive, font, font_scale, thickness
                )
                text_x = (256 - text_width) // 2  # Center text
                text_y = 240  # Near bottom of frame

                # Add black background for text
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + 5),
                    (0, 0, 0),
                    -1,
                )
                # Add text
                cv2.putText(
                    frame,
                    primitive,
                    (text_x, text_y),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                )

                video_writer.write(frame[..., ::-1])  # Convert RGB to BGR for OpenCV

            video_writer.release()

    origin_data_file.close()
    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--action_horizon", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    # Label data files one by one
    data_files = glob.glob(os.path.join(args.libero_dataset_dir, "*.hdf5"))
    results = {}
    for i in tqdm(range(len(data_files))):
        data_path = os.path.join(args.libero_dataset_dir, data_files[i])
        results_json = extract_single_task(data_path, args.action_horizon, args.debug)
        results.update(results_json)

    if args.results_path is None:
        cot_dir = os.path.join(args.libero_dataset_dir, "cot")
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(
            cot_dir, f"primitives_h{args.action_horizon}.json"
        )

    print("Saving results to", args.results_path)
    with open(args.results_path, "w") as f:
        json.dump(results, f)
