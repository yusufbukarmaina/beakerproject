#!/bin/bash

# Beaker Volume Detection - Setup Script for JarvisLab A100
# This script installs all dependencies and sets up the environment

echo "=========================================="
echo "Beaker Volume Detection - Setup"
echo "=========================================="
echo ""

# Check if running on GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: No GPU detected!"
    echo ""
fi

# Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip -q

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install core requirements
echo ""
echo "Installing core dependencies..."
pip install transformers>=4.40.0 -q
pip install datasets>=2.18.0 -q
pip install huggingface_hub>=0.20.0 -q
pip install Pillow>=10.0.0 -q
pip install accelerate>=0.27.0 -q
pip install peft>=0.10.0 -q
pip install bitsandbytes>=0.43.0 -q

# Install Qwen VL utils
echo ""
echo "Installing Qwen VL utils..."
pip install qwen-vl-utils -q

# Install training utilities
echo ""
echo "Installing training utilities..."
pip install tqdm>=4.65.0 -q
pip install tensorboard>=2.15.0 -q

# Install evaluation packages
echo ""
echo "Installing evaluation packages..."
pip install scikit-learn>=1.3.0 -q
pip install numpy>=1.24.0 -q
pip install pandas>=2.0.0 -q
pip install matplotlib>=3.7.0 -q
pip install seaborn>=0.12.0 -q

# Install Gradio
echo ""
echo "Installing Gradio..."
pip install gradio>=4.0.0 -q

# Install flash-attn for faster training
echo ""
echo "Installing flash-attention (this may take a few minutes)..."
pip install flash-attn --no-build-isolation -q

# Create project directories
echo ""
echo "Creating project directories..."
mkdir -p outputs/florence2
mkdir -p outputs/qwen2vl
mkdir -p outputs/test_data/images
mkdir -p outputs/results

echo "Directories created:"
echo "  outputs/florence2/"
echo "  outputs/qwen2vl/"
echo "  outputs/test_data/images/"
echo "  outputs/results/"

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 -c "
import sys

packages = [
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('datasets', 'datasets'),
    ('gradio', 'gradio'),
    ('sklearn', 'scikit-learn'),
    ('PIL', 'Pillow'),
    ('peft', 'peft'),
    ('accelerate', 'accelerate'),
    ('tqdm', 'tqdm'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
]

print('')
all_ok = True
for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'installed')
        print(f'  OK  {display_name} ({version})')
    except ImportError:
        print(f'  FAIL  {display_name} - NOT INSTALLED')
        all_ok = False

import torch
print('')
if torch.cuda.is_available():
    print(f'  OK  CUDA: {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  OK  GPU Memory: {total_mem:.1f} GB')
    print(f'  OK  CUDA Version: {torch.version.cuda}')
else:
    print('  WARN  CUDA not available - training will be slow')

print('')
if all_ok:
    print('All packages installed successfully!')
else:
    print('Some packages failed - check errors above')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Run scripts in this order:"
echo ""
echo "  Step 1 - Test data loading:"
echo "    python3 data_utils.py"
echo ""
echo "  Step 2 - Train Florence-2:"
echo "    python3 train_florence.py"
echo ""
echo "  Step 3 - Train Qwen2-VL:"
echo "    python3 train_qwen.py"
echo ""
echo "  Step 4 - Evaluate both models:"
echo "    python3 evaluate.py"
echo ""
echo "  Step 5 - Launch Gradio demo:"
echo "    python3 demo.py"
echo ""
echo "  Single image inference:"
echo "    python3 inference.py path/to/image.jpg --model florence"
echo "    python3 inference.py path/to/image.jpg --model qwen"
echo ""
