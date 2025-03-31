import argparse
import json
from tqdm import tqdm
import os

# try:
#     hydra.initialize(config_path="../calvin/calvin_models/conf")
# except Exception:
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bboxes_file_path", type=str, default=None)
    parser.add_argument("--reasonings_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data/lzx/embodied-CoT/scripts/generate_embodied_data/new_reasonings")
    args = parser.parse_args()

    bboxes_file_path = args.bboxes_file_path
    with open(bboxes_file_path, "r") as f:
        bboxes = json.load(f)

    reasonings_file_path = args.reasonings_file_path
    with open(reasonings_file_path, "r") as f:
        reasonings = json.load(f)

    new_reasonings = {}

    for file_path in tqdm(reasonings.keys(), desc="Merging"):
        if file_path not in bboxes:
            print(f"File path {file_path} not found in bboxes")
            continue

        bbox = bboxes[file_path]

        try:
            assert len(bbox) == len(reasonings[file_path]["0"]["reasoning"]), f"Length mismatch for {file_path}: {len(bbox)}, {len(reasonings[file_path]['0']['reasoning'])}"
        except Exception as e:
            print(e)
            continue

        new_reasonings[file_path] = reasonings[file_path]

        new_reasonings[file_path]["0"]["features"].update(
            {
                "bboxes": bbox,
            }
        )

    target_dir = os.path.join(args.data_dir, "cot")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Saving to {target_dir}")
    target_file_path = os.path.join(target_dir, f"reasoning.json")

    with open(target_file_path, "w") as f:
        json.dump(new_reasonings, f)
