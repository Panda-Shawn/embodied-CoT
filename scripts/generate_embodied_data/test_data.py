import json


raw_dict = json.load(open("/home/nus/embodied-CoT/scripts/generate_embodied_data/final_reasonings/reasonings_object.json", "r"))

num_skip = 0

num_diff = 0

num_count = 0

for file_name in raw_dict.keys():
    for episode_id in raw_dict[file_name].keys():
        num_count += 1
        if "reasoning" not in raw_dict[file_name][episode_id].keys():
            num_skip += 1
            print("skip ", num_skip)
            continue

        len_reasoning = len(raw_dict[file_name][episode_id]["reasoning"])
        len_gpos = len(raw_dict[file_name][episode_id]["features"]["gripper_position"])
        len_move = len(raw_dict[file_name][episode_id]["features"]["move_primitive"])
        len_box = len(raw_dict[file_name][episode_id]["features"]["bboxes"])

        if len_reasoning != len_gpos or len_reasoning != len_move or len_reasoning != len_box:
            print(f"{num_count}: Episode {episode_id} in {file_name} has different lengths: reasoning {len_reasoning}, gpos {len_gpos}, move {len_move}, box {len_box}")

            num_diff += 1

print(f"num of different lengths: {num_diff} / {num_count}")