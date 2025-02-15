import json
import tensorflow_datasets as tfds
from tqdm import tqdm


def check_if_none_reasoning(episode):
    return len(episode["reasoning"])==0


def check_if_missing_reasoning_steps(episode):
    return len(episode["reasoning"]) != len(episode["features"]["move_primitive"])


if __name__=="__main__":
    dataset_name = "libero_goal_no_noops" # "libero_10_no_noops"
    cat = dataset_name.split("_")[1]
    ds = tfds.load(dataset_name, data_dir="/home/nus/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")

    save_path = f"full_reasonings/new_new_errors_{cat}.json"

    with open(f"full_reasonings/new_new_merged_reasonings_{cat}.json", "r") as reasonings_file:
        reasonings = json.load(reasonings_file)

    errors = {}
    errors_count = 0
    for file_path, task_episodes in tqdm(reasonings.items()):
        for ep_idx, episode in reasonings[file_path].items():
            # print(file_path[:5], ep_idx, type(episode["reasoning"]), len(episode["reasoning"]))
            if check_if_missing_reasoning_steps(episode):
                if file_path not in errors:
                    errors[file_path] = []
                errors[file_path].append(ep_idx)
                errors_count += 1
    meta_error = f"num of content errors: {errors_count} / {len(ds)}"
    errors["meta_error"] = meta_error

    with open(save_path, "w") as out_f:
        json.dump(errors, out_f)