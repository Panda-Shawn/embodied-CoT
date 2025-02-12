import json
import h5py
from tqdm import tqdm
import argparse
import glob
import os


def selected_demo_w_reasoning(data_path, target_data_path, reasonings):
    origin_data_file = h5py.File(data_path, "r")
    origin_data = origin_data_file["data"]
    print(f"Processing {data_path} ...")

    target_data_file = h5py.File(target_data_path, "w")
    target_data = target_data_file.create_group("data")
    for demo in tqdm(origin_data.keys()):
        demo_data = origin_data[demo]
        episode_id = demo_data["episode_id"][()].decode("utf-8")
        print(f"Processing episode {episode_id} ...")

        if episode_id in reasonings.keys():
            origin_data.copy(demo, target_data)
            print(f"Select episode {episode_id} to target data.")

    origin_data_file.close()
    target_data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--libero_target_dir", type=str)
    parser.add_argument("--reasonings_file", type=str)
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.reasonings_file, "r") as f:
        reasonings = json.load(f)

    os.makedirs(args.libero_target_dir, exist_ok=True)

    data_files = glob.glob(os.path.join(args.libero_dataset_dir, "*.hdf5"))
    results = {}
    for i in range(len(data_files)):
        data_path = os.path.join(args.libero_dataset_dir, data_files[i])
        target_data_path = os.path.join(args.libero_target_dir, data_path.split("/")[-1])
        selected_demo_w_reasoning(data_path, target_data_path, reasonings)
    