import tensorflow as tf
import os
from pathlib import Path

class ModifyTFRecordDataset:
    """
    A class to modify existing TFRecord files by adding an 'episode_id' field.
    """
    def __init__(self, original_dataset_dir, modified_dataset_dir):
        self.original_dataset_dir = Path(original_dataset_dir)
        self.modified_dataset_dir = Path(modified_dataset_dir)

    def _add_episode_id(self, serialized_example, episode_id):
        """
        Parse and modify a serialized TFRecord example to add 'episode_id'.
        """
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # Add the 'episode_id' field
        example.features.feature["episode_metadata.episode_id"].bytes_list.value.append(
            f"{episode_id}".encode("utf-8")
        )

        return example.SerializeToString()

    def process_tfrecord(self, input_file, output_file):
        """
        Process a single TFRecord file to add 'episode_id'.
        """
        with tf.io.TFRecordWriter(output_file) as writer:
            for episode_id, serialized_example in enumerate(tf.data.TFRecordDataset(input_file)):
                modified_example = self._add_episode_id(serialized_example.numpy(), episode_id)
                writer.write(modified_example)

    def process_dataset(self):
        """
        Process all TFRecord files in the original dataset directory.
        """
        for root, _, files in os.walk(self.original_dataset_dir):
            for file in files:
                if "tfrecord" in file:
                    input_file = Path(root) / file
                    relative_path = input_file.relative_to(self.original_dataset_dir)
                    output_file = self.modified_dataset_dir / relative_path

                    # Ensure output directory exists
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    print(f"Processing {input_file} -> {output_file}")
                    self.process_tfrecord(str(input_file), str(output_file))

if __name__ == "__main__":
    # Define the paths for the original and modified dataset directories
    original_dataset_dir = "/data/lzx/libero/libero_10_no_noops/1.0.0"
    modified_dataset_dir = "/data/lzx/libero_new/libero_10_no_noops/1.0.0"
    os.makedirs(modified_dataset_dir, exist_ok=True)

    # Create an instance of ModifyTFRecordDataset and process the dataset
    modifier = ModifyTFRecordDataset(original_dataset_dir, modified_dataset_dir)
    modifier.process_dataset()

    print(f"Modified dataset saved to {modified_dataset_dir}.")