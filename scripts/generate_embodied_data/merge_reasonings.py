import json
import tensorflow_datasets as tfds
from tqdm import tqdm


if __name__=="__main__":
    dataset_name = "libero_spatial_no_noops" # "libero_10_no_noops"
    ds = tfds.load(dataset_name, data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")

    save_path = "full_reasonings/final_reasonings.json"

    with open("bounding_boxes/bboxes/full_bboxes.json", "r") as bboxes_file:
        bboxes = json.load(bboxes_file)

    with open("gripper_positions/gripper_positions/gripper_positions.json", "r") as gripper_positions_file:
        gripper_positions = json.load(gripper_positions_file)

    with open("full_reasonings/reasonings.json", "r") as reasonings_file:
        reasonings = json.load(reasonings_file)

    gripper_positions_json = {}
    for file_path, task_episodes in tqdm(reasonings.items()):
        for ep_idx, episode in reasonings[file_path].items():
            reasonings[file_path][ep_idx]["features"]["bboxes"] = bboxes[file_path][ep_idx]["bboxes"]
            reasonings[file_path][ep_idx]["features"]["gripper_position"] = gripper_positions[file_path][ep_idx]


    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f)


