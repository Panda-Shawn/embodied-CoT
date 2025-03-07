import argparse
import numpy as np
import h5py
import os
import json
import glob
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation


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
    diff = move[-1] - move[0]

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
    origin_data_file = open(data_path, "rb")
    origin_data = pickle.load(origin_data_file)
    results_json = {}
    print(f"Processing {data_path} ...")

    episode_ee_trans = [origin_data["observations"][i]["robot_state"]["ee_pos"] for i in range(len(origin_data["observations"]))]
    episode_ee_rots = [Rotation.from_quat(origin_data["observations"][i]["robot_state"]["ee_quat"]).as_euler("xyz") for i in range(len(origin_data["observations"]))]
    episode_action = np.array(origin_data["actions"])
    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    gripper_action = episode_action[:, -1:]

    episode_states = np.concatenate([episode_ee_trans[:-1], episode_ee_rots[:-1], gripper_action], axis=-1)

    move_trajs = [
        episode_states[i : i + action_horizon]
        for i in range(len(episode_states) - 1)
    ]
    primitives_list = [classify_movement(move)[0] for move in move_trajs]
    primitives_list.append(primitives_list[-1])

    results_json[data_path] = primitives_list
    if debug:
        pass

    origin_data_file.close()
    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--action_horizon", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    # Label data files one by one
    data_files = glob.glob(os.path.join(args.dataset_dir, "*/*.pkl"))
    results = {}
    for i in tqdm(range(len(data_files))):
        data_path = os.path.join(args.dataset_dir, data_files[i])
        results_json = extract_single_task(data_path, args.action_horizon, args.debug)
        results.update(results_json)

    if args.results_path is None:
        cot_dir = os.path.join(args.dataset_dir, "cot")
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(
            cot_dir, f"primitives_h{args.action_horizon}.json"
        )

    print("Saving results to", args.results_path)
    with open(args.results_path, "w") as f:
        json.dump(results, f)
