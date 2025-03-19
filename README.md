# Robotic Control via Embodied Chain-of-Thought Reasoning

## Installation

First, create a virtual environment with `python==3.10` and `torch`.

```bash
# Create and activate conda environment
conda create -n labelcot python=3.10 -y
conda activate labelcot

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!
```

Then, follow the official instruction to install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). After that, continue to install this cot labeling library from source.

```bash
# Clone and install the openvla repo
git clone https://github.com/Panda-Shawn/embodied-CoT.git
cd embodied-CoT
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

Finally, use `pip install -r requirements-modified.txt` to install some extra libraries for data generation pipeline.
