import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import (
    NumpyFloatValuesEncoder,
    decode_instance_names,
)
import glob

from prismatic import load

from env_utils import process_single_image, INSTANCE_ID_TO_NAMES, LANGUALGE_INSTRUCTIONS

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
try:
    hf_token = Path("./.hf_token").read_text().strip()
except FileNotFoundError:
    hf_token = None

import pickle


if __name__ == "__main__":
    import sys
    print("Command-line arguments:", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # Label data files one by one
    data_files = glob.glob(os.path.join(args.dataset_dir, "*/*.pkl"))
    results = {}
    num_errors = 0
    for i in tqdm(range(len(data_files))):
        data_path = os.path.join(args.dataset_dir, data_files[i])
        try:
            data = pickle.load(open(data_path, "rb"))
        except Exception as e:
            num_errors += 1
            print(f"Error loading {data_path}: {e}, num: {num_errors}")
            continue