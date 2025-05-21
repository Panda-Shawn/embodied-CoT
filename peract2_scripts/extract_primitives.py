import argparse
import numpy as np
import h5py
import os
import json
import glob
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation
import tensorflow_datasets as tfds


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


def convert_gripper_action(action):
    target_action = []
    current_state = -1
    for i in range(len(action)):
        if action[i] == -1:
            current_state = 1
        elif action[i] == 1:
            current_state = -1
        
        target_action.append(current_state)
    return target_action


def extract_single_task(data_path, episode, action_horizon, debug: bool = False):
    results_json = {}
    print(f"Processing {data_path} ...")
    
    # right arm
    episode_ee_trans_right = []
    episode_ee_rots_right = []
    episode_action_right = []
    for step in episode["steps"]:
        episode_ee_trans_right.append(step["observation"]["right_gripper_pose"].numpy()[:3])
        episode_ee_rots_right.append(Rotation.from_quat(step["observation"]["right_gripper_pose"].numpy()[3:]).as_euler("xyz"))
        episode_action_right.append(step["right_action"].numpy()[-1])
    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    right_gripper_action = convert_gripper_action(episode_action_right)
    right_gripper_action = np.array(right_gripper_action).reshape(-1, 1)

    episode_states_right = np.concatenate([episode_ee_trans_right[:-1], episode_ee_rots_right[:-1], right_gripper_action[:-1]], axis=-1)

    move_trajs_right = [
        episode_states_right[i : i + action_horizon]
        for i in range(len(episode_states_right) - 1)
    ]
    
    # left arm
    episode_ee_trans_left = []
    episode_ee_rots_left = []
    episode_action_left = []
    for step in episode["steps"]:
        episode_ee_trans_left.append(step["observation"]["left_gripper_pose"].numpy()[:3])
        episode_ee_rots_left.append(Rotation.from_quat(step["observation"]["left_gripper_pose"].numpy()[3:]).as_euler("xyz"))
        episode_action_left.append(step["left_action"].numpy()[-1])
    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    left_gripper_action = convert_gripper_action(episode_action_left)
    left_gripper_action = np.array(left_gripper_action).reshape(-1, 1)
    episode_states_left = np.concatenate([episode_ee_trans_left[:-1], episode_ee_rots_left[:-1], left_gripper_action[:-1]], axis=-1)
    move_trajs_left = [
        episode_states_left[i : i + action_horizon]
        for i in range(len(episode_states_left) - 1)
    ]
    
    primitives_list = [classify_movement(move_left)[0] + "; " + classify_movement(move_right)[0] for move_left, move_right in zip(move_trajs_left, move_trajs_right)]
    primitives_list.append(primitives_list[-1])

    results_json[data_path] = primitives_list
    if debug:
        pass

    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--action_horizon", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    ds = tfds.load(
        args.dataset_dir.split("/")[-1].split("_dir")[0],
        data_dir=args.dataset_dir,
        split=f"train[{0}%:{100}%]",
    )

    results = {}
    for episode in tqdm(ds):
        data_path = episode["episode_metadata"]["file_path"].numpy().decode()
        results_json = extract_single_task(data_path, episode, args.action_horizon, args.debug)
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
