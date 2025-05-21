import tensorflow_datasets as tfds
import numpy as np


ds = tfds.load(
    "colosseum_dataset",
    data_dir="/data2/lzixuan/colosseum_dir",
    split=f"train[{0}%:{100}%]",
)


# for episode in ds:
#     for step in episode["steps"]:
#         print(step["observation"]["gripper_open"].numpy(), step["action"].numpy()[-1], step["observation"]["gripper_joint_positions"].numpy())
#     break

for i, episode in enumerate(ds):
    if i == 1:
        for step in episode["steps"]:
            mask = step["observation"]["front_mask"].numpy()
            print(np.unique(mask))
        break