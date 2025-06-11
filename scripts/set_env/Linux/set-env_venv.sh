#!/bin/bash

# Create a Python virtual environment
echo "Creating Python virtual environment in .venv..."
python3.10 -m venv .venv || {
    echo "Python 3.10 not found. Please install it first."
    exit 1
}

# Activate the virtual environment
source .venv/bin/activate

echo -e "\n[Upgrading pip] ====================================================="
python -m pip install --upgrade pip

# Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "\n[Installing packages from requirements.txt] ========================="
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping..."
fi

# Install PyTorch with CUDA 11.7
echo -e "\n[Installing PyTorch with CUDA 11.7] ================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo -e "\nVirtual environment setup complete!"
echo "To activate this environment in the future, run: source .venv/bin/activate"
echo "To deactivate the environment, simply run: deactivate"
