import json

import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os


def show_box(box, ax, text, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, f"{text}", color="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="/data/lzx/tensorflow_datasets")
    parser.add_argument("--reasoning_dir", type=str, default="/data/lzx/embodied-CoT/scripts/generate_embodied_data/new_reasonings/final_reasonings")
    args = parser.parse_args()

    reasoning_file_path = os.path.join(args.reasoning_dir, f"reasoning_{args.libero_task_suite}.json")
    with open(reasoning_file_path, "r") as f:
        reasonings = json.load(f)

    ds = tfds.load(args.libero_task_suite, data_dir=args.data_dir, split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")

    for ep_idx, episode in enumerate(ds):

        episode_id = episode["episode_metadata"]["episode_id"].numpy().decode()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")
        # if episode_id != 1:
        #     continue

        for step_idx, step in enumerate(episode["steps"]):
            # Load the image
            image = step["observation"]["image"].numpy()
            
            # Extract the caption and bounding boxes
            caption = reasonings[file_path][str(episode_id)]["reasoning"][str(step_idx)]["task"]
            bboxes = reasonings[file_path][str(episode_id)]["features"]["bboxes"][step_idx]

            # Plot the image using matplotlib
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

            # Draw bounding boxes
            for text, bbox in bboxes:
                show_box(bbox, ax, text, "red")

            # Add caption as a title
            # Add caption as multiline text
            caption_text = "\n".join([caption[i:i+70] for i in range(0, len(caption), 70)])
            fig.text(0.5, 0.01, caption_text, ha="center", fontsize=12, color="white", backgroundcolor="black")

            # Save the image
            output_path = f"vis_bboxes/output_ep_{episode_id}_step_{step_idx}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved image with bounding boxes and caption to {output_path}")
            # break
        break
