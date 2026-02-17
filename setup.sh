#!/bin/bash

# Beaker Volume Detection - Setup Script for JarvisLab A100
# Fixed version - explicit installs with error output visible

echo "=========================================="
echo "Beaker Volume Detection - Setup"
echo "=========================================="
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Update pip
echo "--- Updating pip ---"
python3 -m pip install --upgrade pip

echo ""
echo "--- Installing PyTorch (CUDA 12.6) ---"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "--- Installing transformers ---"
pip install "transformers>=4.40.0"

echo ""
echo "--- Installing peft ---"
pip install "peft>=0.10.0"

echo ""
echo "--- Installing bitsandbytes ---"
pip install bitsandbytes

echo ""
echo "--- Installing huggingface_hub ---"
pip install "huggingface_hub>=0.20.0"

echo ""
echo "--- Installing accelerate ---"
pip install "accelerate>=0.27.0"

echo ""
echo "--- Installing datasets ---"
pip install "datasets>=2.18.0"

echo ""
echo "--- Installing Qwen VL utils ---"
pip install qwen-vl-utils

echo ""
echo "--- Installing training utilities ---"
pip install tensorboard

echo ""
echo "--- Installing Gradio ---"
pip install "gradio>=4.0.0"

echo ""
echo "--- Installing evaluation and utility packages ---"
pip install scikit-learn numpy pandas matplotlib seaborn tqdm Pillow python-dotenv

echo ""
echo "--- Installing flash-attention (may take a few minutes) ---"
pip install flash-attn --no-build-isolation

# Create project directories
echo ""
echo "--- Creating project directories ---"
mkdir -p outputs/florence2
mkdir -p outputs/qwen2vl
mkdir -p outputs/test_data/images
mkdir -p outputs/results
echo "Done."

# Verify all installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 -c "
packages = [
    ('torch',          'torch'),
    ('transformers',   'transformers'),
    ('datasets',       'datasets'),
    ('gradio',         'gradio'),
    ('sklearn',        'scikit-learn'),
    ('PIL',            'Pillow'),
    ('peft',           'peft'),
    ('accelerate',     'accelerate'),
    ('huggingface_hub','huggingface_hub'),
    ('tqdm',           'tqdm'),
    ('pandas',         'pandas'),
    ('numpy',          'numpy'),
    ('matplotlib',     'matplotlib'),
    ('seaborn',        'seaborn'),
    ('tensorboard',    'tensorboard'),
]

print('')
all_ok = True
for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'installed')
        print(f'  OK    {display_name:<20} ({version})')
    except ImportError:
        print(f'  FAIL  {display_name:<20} NOT INSTALLED')
        all_ok = False

import torch
print('')
if torch.cuda.is_available():
    print(f'  OK    CUDA device    : {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  OK    GPU Memory     : {total_mem:.1f} GB')
    print(f'  OK    CUDA Version   : {torch.version.cuda}')
else:
    print('  WARN  CUDA not available')

print('')
if all_ok:
    print('All packages installed successfully!')
else:
    print('Some packages failed. Re-run the failed ones manually.')
    print('Example:  pip install transformers peft')
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