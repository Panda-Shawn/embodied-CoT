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
    "lamp": {
        26: "lamp_base",
        27: "lamp_bulb",
        28: "lamp_hood",
    },
    "one_leg": {
        26: "square_table_top",
        27: "square_table_leg_1",
        28: "square_table_leg_2",
        29: "square_table_leg_3",
        30: "square_table_leg_4",
    },
    "round_table": {
        26: "round_table_top",
        27: "round_table_leg",
        28: "round_table_base",
    },
    "cabinet": {
        26: "cabinet_body",
        27: "cabinet_door_1",
        28: "cabinet_door_2",
        29: "cabinet_top", 
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
