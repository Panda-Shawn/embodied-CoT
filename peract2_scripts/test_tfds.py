import tensorflow_datasets as tfds
import numpy as np


ds = tfds.load(
    "handover_item",
    data_dir="/data/lzx/peract2/handover_item_dir",
    split=f"train[{0}%:{100}%]",
)

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


# for i, episode in enumerate(ds):
#     if i == 1:
#         gripper_action = []
#         for step in episode["steps"]:
#             print(step["observation"]["right_gripper_open"].numpy(), step["right_action"].numpy()[-1], step["observation"]["right_gripper_joint_positions"].numpy())
#             gripper_action.append(step["right_action"].numpy()[-1])
#         print(gripper_action)
#         gripper_action = convert_gripper_action(gripper_action)
#         print(gripper_action)
#         break

for i, episode in enumerate(ds):
    if i == 1:
        for step in episode["steps"]:
            mask = step["observation"]["front_mask"].numpy()
            print(np.unique(mask))
            break
        break
