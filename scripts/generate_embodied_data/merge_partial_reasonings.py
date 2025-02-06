import json
import tensorflow_datasets as tfds
from tqdm import tqdm


if __name__=="__main__":
    dataset_name = "libero_object_no_noops" # "libero_10_no_noops"
    cat = dataset_name.split("_")[1]
    ds = tfds.load(dataset_name, data_dir="/home/nus/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")

    save_path = f"full_reasonings/new_new_merged_reasonings_{cat}.json"

    with open(f"full_reasonings/new_merged_reasonings_{cat}.json", "r") as reasonings_file:
        reasonings = json.load(reasonings_file)
    
    with open(f"requery_reasonings/reasonings_newprompt_{cat}.json", "r") as reasonings_file:
        error_reasonings = json.load(reasonings_file)

    merged_reasonings = {}
    for file_path, task_episodes in tqdm(reasonings.items()):
        for ep_idx, episode in reasonings[file_path].items():
            if file_path not in merged_reasonings.keys():
                merged_reasonings[file_path] = {}
            merged_reasonings[file_path][ep_idx] = episode
            if error_reasonings[file_path][ep_idx]["reasoning"] is not None and len(error_reasonings[file_path][ep_idx]["reasoning"]) != 0:
                merged_reasonings[file_path][ep_idx] = error_reasonings[file_path][ep_idx]



    with open(save_path, "w") as out_f:
        json.dump(merged_reasonings, out_f)


