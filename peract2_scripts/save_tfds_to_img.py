import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import os


task_name = "lift_ball"

ds = tfds.load(
    task_name,
    data_dir=f"/data/lzx/peract2/{task_name}_dir",
    split=f"train[{0}%:{100}%]",
)

output_dir = f"{task_name}/traj_imgs"
os.makedirs(output_dir, exist_ok=True)

for i, episode in enumerate(ds):
    if i == 1:
        print("file path: ", episode["episode_metadata"]["file_path"].numpy().decode())
        for j, step in enumerate(episode["steps"]):
            lang_instruction = step["language_instruction"].numpy().decode()
            print(lang_instruction)
            image = Image.fromarray(step["observation"]["front_image"].numpy())
            output_path = os.path.join(output_dir, f"img_{j}.png")
            image.save(output_path)
        break