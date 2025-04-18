import tensorflow_datasets as tfds
from tqdm import tqdm


ds = tfds.load('libero_90', data_dir="/data2/lzixuan/Fur_cot", split='train')
# ds = tfds.load('libero_10', data_dir="/data2/lzixuan/Libero/end_to_end", split='train')


for episode in tqdm(ds):
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print("file_path:", file_path)
    break