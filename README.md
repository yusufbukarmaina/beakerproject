# 🧪 Beaker Volume Detection using Vision-Language Models

This project uses fine-tuned Vision-Language Models (Florence-2 and Qwen2-VL) to detect beaker capacity and liquid volume from images.

## 📋 Project Overview

- **Dataset**: 2000+ beaker images from Hugging Face (`yusufbukarmaina/Beakers1`)
- **Beaker Types**: 100mL and 250mL
- **Viewpoints**: Left, Right, Above, Below, Front
- **Backgrounds**: Controlled and Uncontrolled
- **Models**: Florence-2 and Qwen2-VL (fine-tuned with LoRA)
- **Hardware**: Optimized for JarvisLab A100 (40GB)

## 🚀 Quick Start

### 1. Installation
```bash
chmod +x setup.sh
./setup.sh
```

Or manually install:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python data_utils.py
```

This will:
- Load the dataset from Hugging Face
- Split into train (70%), validation (15%), test (15%)
- Save test data to `outputs/test_data/`

### 3. Training

**Train Florence-2:**
```bash
python train_florence.py
```

**Train Qwen2-VL:**
```bash
python train_qwen.py
```

Models will be saved to:
- `outputs/florence2/final_model/`
- `outputs/qwen2vl/final_model/`

### 4. Evaluation
```bash
python evaluate.py
```

This generates:
- MAE, RMSE, R² metrics for both models
- Comparison visualizations
- Performance analysis by viewpoint and background

Results saved to `outputs/results/`

### 5. Demo

**Launch Gradio Interface:**
```bash
python demo.py
```

**Single Image Inference:**
```bash
python inference.py path/to/image.jpg --model florence
python inference.py path/to/image.jpg --model qwen
```

## 📁 Project Structure
```
beaker_project/
├── config.py                 # Configuration settings
├── data_utils.py            # Data loading and processing
├── train_florence.py        # Florence-2 training script
├── train_qwen.py            # Qwen2-VL training script
├── evaluate.py              # Model evaluation
├── demo.py                  # Gradio interface
├── inference.py             # Single image inference
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup script
├── README.md               # This file
└── outputs/
    ├── florence2/          # Florence-2 checkpoints
    ├── qwen2vl/           # Qwen2-VL checkpoints
    ├── test_data/         # Test images and metadata
    └── results/           # Evaluation results and plots
```

## 🔧 Configuration

Edit `config.py` to customize:
- Batch sizes
- Learning rates
- Number of epochs
- LoRA parameters
- Output directories

**Default settings for A100 40GB:**
- Florence-2: batch_size=4, gradient_accumulation=4
- Qwen2-VL: batch_size=2, gradient_accumulation=8

## 📊 Evaluation Metrics

Both models are evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error in mL
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (R-squared): Goodness of fit (0 to 1, higher is better)

Additional metrics:
- Beaker capacity classification accuracy
- Performance by viewpoint
- Performance by background type

## 🎯 Expected Performance

Target metrics (will vary based on training):
- MAE: < 5 mL
- RMSE: < 10 mL
- R²: > 0.95
- Beaker Accuracy: > 98%

## 💾 Test Data Download

After running `data_utils.py` or any training script, test data is saved to:
```
outputs/test_data/
├── images/          # Test images
├── metadata.json    # Image metadata
└── README.txt       # Dataset info
```

You can download this folder for demo purposes.

## 🐛 Troubleshooting

**Out of Memory Error:**
- Reduce batch size in `config.py`
- Increase gradient accumulation steps
- Use smaller LoRA rank

**Model Loading Issues:**
- Ensure models are trained before evaluation/demo
- Check model paths in `config.py`
- Verify HuggingFace cache has space

**Dataset Loading Issues:**
- Check internet connection
- Verify HuggingFace dataset name
- Clear cache: `rm -rf ~/.cache/huggingface/`

## 📝 Citation

If you use this code, please cite:
```bibtex
@misc{beaker-volume-detection,
  author = {Your Name},
  title = {Beaker Volume Detection using Vision-Language Models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/beaker-volume-detection}
}
```

## 📄 License

MIT License - feel free to use for research and commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]
```

### **11. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Model outputs
outputs/
*.pt
*.pth
*.bin
*.safetensors

# Logs
*.log
logs/
tensorboard/
wandb/

# Data
data/
*.csv
*.json
!requirements.txt
!config.py

# OS
.DS_Store
Thumbs.db

# HuggingFace cache
.cache/