import argparse
import hydra
import numpy as np
import json
from tqdm import tqdm

# try:
#     hydra.initialize(config_path="../calvin/calvin_models/conf")
# except Exception:
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data2/zbzhu/task_D_D")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--action_horizon", type=int, default=10)
    args = parser.parse_args()

    config = hydra.compose(
        config_name="lang_ann",
        overrides=[
            f"datamodule.root_data_dir={args.data_dir}",
            "datamodule/observation_space=lang_rgb_static_robot_scene_abs_act",
        ],
    )

    data_module = hydra.utils.instantiate(config.datamodule, num_workers=4)
    data_module.prepare_data()
    data_module.setup()

    if args.data_split == "train":
        dataset = data_module.train_datasets["vis"]
    elif args.data_split == "val":
        dataset = data_module.val_datasets["vis"]
    else:
        raise ValueError(f"Invalid data split: {args.data_split}")

    file_name = dataset.abs_datasets_dir / config.lang_folder / "auto_lang_ann.npy"
    lang_anns = np.load(file_name, allow_pickle=True).reshape(-1)[0]

    plan_subtasks_dir = (
        dataset.abs_datasets_dir
        / "cot"
        / f"plan_subtasks_batched_h{args.action_horizon}"
    )

    merged_plan_subtasks = {}
    tqdm_bar = tqdm(
        total=len(lang_anns["info"]["indx"]),
        desc=f"Merging plan subtasks (h={args.action_horizon})",
    )
    for lang_idx, (ep_start, ep_end) in enumerate(lang_anns["info"]["indx"]):
        ep_start, ep_end = int(ep_start), int(ep_end)
        merged_plan_subtasks[str(ep_start)] = {}
        plan_subtasks_file_name = plan_subtasks_dir / f"{ep_start}_{ep_end}.json"
        with open(plan_subtasks_file_name, "r") as f:
            plan_subtasks = json.load(f)

        for step in plan_subtasks.keys():
            if str(int(step) + ep_start) in merged_plan_subtasks[str(ep_start)]:
                print(f"Duplicate step {step} for episode {ep_start}_{ep_end}")
            merged_plan_subtasks[str(ep_start)][str(int(step) + ep_start)] = (
                plan_subtasks[step]
            )

        tqdm_bar.update(1)

    with open(
        plan_subtasks_dir.parent / f"merged_cot_h{args.action_horizon}.json", "w"
    ) as f:
        json.dump(merged_plan_subtasks, f)