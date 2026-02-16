#!/bin/bash

# Beaker Volume Detection - Setup Script for JarvisLab A100
# This script installs all dependencies and sets up the environment

echo "=========================================="
echo "Beaker Volume Detection - Setup"
echo "=========================================="
echo ""

# Check if running on GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠ WARNING: No GPU detected!"
    echo ""
fi

# Update pip
echo "Updating pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install additional packages
echo ""
echo "Installing additional packages..."
pip install flash-attn --no-build-isolation

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 << EOF
import sys

packages = [
    'torch',
    'transformers',
    'datasets',
    'gradio',
    'sklearn',
    'PIL',
    'peft',
    'accelerate'
]

print("\nInstalled packages:")
for pkg in packages:
    try:
        if pkg == 'PIL':
            __import__('PIL')
            print(f"✓ {pkg}")
        elif pkg == 'sklearn':
            __import__('sklearn')
            print(f"✓ {pkg}")
        else:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {pkg} (v{version})")
    except ImportError:
        print(f"✗ {pkg} - NOT INSTALLED")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n✗ CUDA not available")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python data_utils.py (to test data loading)"
echo "2. Run: python train_florence.py (to train Florence-2)"
echo "3. Run: python train_qwen.py (to train Qwen2-VL)"
echo "4. Run: python evaluate.py (to evaluate both models)"
echo "5. Run: python demo.py (to launch Gradio demo)"
echo ""
echo "For single image inference:"
echo "  python inference.py <image_path> --model florence"
echo "  python inference.py <image_path> --model qwen"
echo ""