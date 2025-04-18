import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from einops import rearrange
import tensorflow_datasets as tfds
from tqdm import tqdm
import h5py
import os


def extract_radio_feature(image, image_processor, model):
    patch_size = 16
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to("cuda")
    summary, features = model(pixel_values)
    spatial_features = rearrange(features, 'b (h w) d -> b d h w', h=pixel_values.shape[-2] // patch_size, w=pixel_values.shape[-1] // patch_size)
    return features.cpu().detach(), spatial_features.cpu().detach()


# def clean_path(path):
#     """Convert file paths into HDF5-safe group names"""
#     return path.replace("/", "_").replace("\\", "_")


def main():
    # Set model
    hf_repo = "nvidia/RADIO"  # You can change this to other variants like RADIO-B, RADIO-L, etc.
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
    model.eval().to("cuda")

    # Load LIBERO dataset
    ds = tfds.load('libero_10', data_dir="/home/nus/Libero/end_to_end", split='train')

    # Prepare HDF5 file
    hdf5_path = "radio_embedding.h5"
    h5f = h5py.File(hdf5_path, "w")
    print(f"Saving embeddings to {hdf5_path}")

    pbar = tqdm(total=len(ds))
    pbar.set_description("Extracting features")

    for ep_idx, episode in enumerate(ds):
        # if ep_idx == 10:
        #     break

        episode_id = episode["episode_metadata"]["episode_id"].numpy().decode()
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        print(f"Processing episode: {episode_id}, path: {file_path}")

        episode_features = []

        for step_idx, step in enumerate(episode["steps"]):
            image = step["observation"]["image"].numpy()
            image = Image.fromarray(image).convert('RGB').resize((224, 224))

            features, _ = extract_radio_feature(image, image_processor, model)  # (1, 196, 1280)
            episode_features.append(features)

        # Stack to shape [T, 196, 1280]
        feature_tensor = torch.cat(episode_features, dim=0).to(torch.float32)

        # Save to HDF5
        # group_name = clean_path(file_path)
        h5f.create_dataset(file_path, data=feature_tensor.numpy(), compression="gzip")

        pbar.update(1)

    pbar.close()
    h5f.close()
    print(f"All features saved to {hdf5_path}")


if __name__ == "__main__":
    main()
