#!/bin/bash

# Define here the path to your conda installation
# Usually it's in the home directory
CONDAPATH="$HOME/miniconda3"

# Source conda
. "$CONDAPATH/etc/profile.d/conda.sh"

# Create conda environment with Python 3.8 for DNABERT
echo "Creating conda environment 'dnabert' with Python 3.8..."
conda create --name dnabert python=3.8 -y

# Activate the environment
conda activate dnabert

echo -e "\n[Installing required packages] ========================================="
python -m pip install --upgrade pip

# Install specific versions of packages for DNABERT compatibility
pip install einops==0.6.1
pip install peft==0.4.0
pip install huggingface-hub==0.16.4
pip install scikit-learn
pip install matplotlib
pip install progressbar
pip install tensorboard==2.13.0
pip install tensorboard-data-server==0.7.1

# Install PyTorch with CUDA 11.7
echo -e "\n[Installing PyTorch with CUDA 11.7] ================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo -e "\nDNABERT environment setup complete! You can now use the 'dnabert' environment."
echo "To activate it in the future, run: conda activate dnabert"
