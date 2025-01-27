import json

import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt


def show_box(box, ax, score, text, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, f"{score:.3f}  {text}", color="white")


def plot_bboxes():
    ds = tfds.load("libero_spatial_no_noops", data_dir="/data/lzx/libero_new", split=f"train[{0}%:{100}%]")
    print(f"data size: {len(ds)}")
    print("Done.")

    for ep_idx, episode in enumerate(ds):

        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"starting ep: {episode_id}, {file_path}")

        with open("bounding_boxes/descriptions/captions.json", "r") as captions_file:
            captions_dict = json.load(captions_file)

        with open("bounding_boxes/bboxes/full_bboxes.json", "r") as bboxes_file:
            bboxes_dict = json.load(bboxes_file)

        if file_path not in bboxes_dict.keys() or str(episode_id) not in bboxes_dict[file_path].keys():
            print(f"File path {file_path} and episode id {episode_id} not found in full_bboxes.json")
            continue

        for step_idx, step in enumerate(episode["steps"]):
            # Load the image
            image = step["observation"]["image"].numpy()
            
            # Extract the caption and bounding boxes
            caption = captions_dict[file_path][str(episode_id)]["caption"]
            bboxes = bboxes_dict[file_path][str(episode_id)]["bboxes"][step_idx]

            # Plot the image using matplotlib
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

            # Draw bounding boxes
            for score, text, bbox in bboxes:
                show_box(bbox, ax, score, text, "red")

            # Add caption as a title
            # Add caption as multiline text
            caption_text = "\n".join([caption[i:i+70] for i in range(0, len(caption), 70)])
            fig.text(0.5, 0.01, caption_text, ha="center", fontsize=12, color="white", backgroundcolor="black")

            # Save the image
            output_path = f"vis_bboxes/output_ep_{episode_id}_step_{step_idx}.png"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved image with bounding boxes and caption to {output_path}")
            break


if __name__ == "__main__":
    
    plot_bboxes()
