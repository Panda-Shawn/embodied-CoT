"""
Utils for evaluating policies in LIBERO simulation environments.
Copy from OpenVLA.
"""

import math
import os
from datetime import date, datetime

import imageio
import numpy as np
import tensorflow as tf


INSTANCE_ID_TO_NAMES = {
    "basketball_in_hoop": {
        83: "basketball_hoop",
        87: "basketball",
    },
    "close_box": {
        81: "box_base",
        88: "box_lid",
    },
    "close_laptop_lid": {
        82: "laptop_base",
        85: "laptop_lid",
    },
    "empty_dishwasher": {
        85: "dishwasher_door",
        88: "dishwasher_tray",
        94: "dishwasher_plate",
        98: "dishwasher_base",
    },
    "get_ice_from_fridge": {
        87: "fridge",
        90: "plane",
        92: "cup",
    },
    "hockey": {
        81: "hockey_goal",
        84: "hockey_stick",
        88: "hockey_ball",
    },
    "insert_onto_square_peg": {
        81: "square_base",
        82: "pillar0",
        83: "pillar1",
        84: "pillar2",
        96: "square_ring",
    },
    "meat_on_grill": {
        83: "chicken",
        84: "grill",
        89: "steak",
    },
    "move_hanger": {
        81: "clothes_rack",
        87: "clothes_rack0",
        91: "clothes_hanger",
    },
    "open_drawer": {
        81: "drawer_frame",
        83: "drawer_bottom",
        90: "drawer_middle",
        93: "drawer_top",
    },
    "place_wine_at_rack_location": {
        83: "rack_bottom",
        85: "rack_top",
        97: "wine_bottle",
    },
    "put_money_in_safe": {
        81: "safe_body",
        83: "dollar_stack",
        89: "safe_door",
    },
    "reach_and_drag": {
        81: "cube",
        88: "stick",
    },
    "scoop_with_spatula": {
        81: "cuboid",
        88: "spatula",
    },
    "setup_chess": {
        83: "white_squares",
        84: "black_squares",
        85: "chess_board_base",
        128: "white_queenside_knight",
        130: "white_kingside_knight", 
        133: "white_pawn_a", 
        135: "white_pawn_b", 
        137: "white_pawn_c",
        139: "white_pawn_d",
        141: "white_pawn_e",
        143: "white_pawn_f",
        145: "white_pawn_g",
        147: "white_pawn_h",
        150: "white_queenside_rook", 
        152: "white_kingside_rook", 
        155: "white_queenside_bishop",
        157: "white_kingside_bishop",
        160: "white_queen",
        163: "white_king",
        167: "black_queenside_knight",
        169: "black_kingside_knight",
        172: "black_pawn_f",
        174: "black_pawn_a",
        176: "black_pawn_b",
        178: "black_pawn_c",
        180: "black_pawn_d",
        182: "black_pawn_e",
        184: "black_pawn_g",
        186: "black_pawn_h",
        189: "black_queenside_rook",
        191: "black_kingside_rook",
        194: "black_queenside_bishop",
        196: "black_kingside_bishop",
        199: "black_king",
        202: "black_queen",
    },
    "slide_block_to_target": {
        81: "target",
        83: "block",
    },
    "stack_cups": {
        82: "cup1",
        87: "cup2",
        99: "cup3",
    },
    "straighten_rope": {
        81: "tail",
        115: "head",
        123: "head_target",
        125: "head_tail",
    },
    "turn_oven_on": {
        83: "oven_frame",
        85: "oven_stove_top0",
        86: "oven_gas_source0",
        89: "oven_knob_12",
        90: "oven_knob_13",
        91: "oven_knob_14",
        95: "oven_door",
        98: "oven_knob_9",
    },
    "wipe_desk": {
        90: "sponge",
    },
}

LANGUALGE_INSTRUCTIONS = {
    "lamp": "Assemble a lamp with a lamp base, a lamp bulb, and a lamp hood.",
    "one_leg": "Assembly one square table leg onto a square table top.",
    "round_table": "Assemble a round table with a round table top, a round table leg, and a round table base.",
    "cabinet": "Assembly a cabinet with a cabinet body, a cabinet top, and two cabinet doors.",
}

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def process_single_image(img, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = img[::-1, ::-1].copy()
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{date.today()}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{datetime.now()}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
