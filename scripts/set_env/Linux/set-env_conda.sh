#!/bin/bash

# Define here the path to your conda installation
# Usually it's in the home directory, but you might need to change it
CONDAPATH="$HOME/miniconda3"

# Check if conda path exists
if [ ! -d "$CONDAPATH" ]; then
    echo "Error: Conda installation not found at $CONDAPATH"
    echo "Please modify the CONDAPATH in this script to point to your conda installation directory."
    echo "Common locations include:"
    echo "  - $HOME/miniconda3"
    echo "  - $HOME/anaconda3"
    echo "  - /opt/conda"
    echo "  - /usr/local/miniconda3"
    echo "  - /usr/local/anaconda3"
    exit 1
fi

# Source conda
. "$CONDAPATH/etc/profile.d/conda.sh"

# Create conda environment with Python 3.10
echo "Creating conda environment 'benchmark' with Python 3.10..."
conda create --name benchmark python=3.10 -y

# Activate the environment
conda activate benchmark

echo -e "\n[Install packages in requirements.txt] ======================================="
python -m pip install --upgrade pip

# Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping..."
fi

# Install PyTorch with CUDA 11.7
echo -e "\n[Installing PyTorch with CUDA 11.7] ================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo -e "\nEnvironment setup complete! You can now use the 'benchmark' environment."
echo "To activate it in the future, run: conda activate benchmark"
