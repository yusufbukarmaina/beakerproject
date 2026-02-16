"""
Configuration file for Beaker Volume Detection Project
"""
import os
from pathlib import Path

# Dataset Configuration
DATASET_NAME = "yusufbukarmaina/Beakers1"
DATASET_SPLIT_SEED = 42

# Train/Val/Test Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model Configurations
FLORENCE_MODEL_NAME = "microsoft/Florence-2-base"
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Training Hyperparameters - Florence-2
FLORENCE_CONFIG = {
    "batch_size": 4,  # Optimized for 40GB A100
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "num_epochs": 20,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "fp16": True,
    "dataloader_num_workers": 4,
}

# Training Hyperparameters - Qwen2-VL
QWEN_CONFIG = {
    "batch_size": 2,  # Smaller batch for VLM
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-6,
    "num_epochs": 15,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "fp16": True,
    "dataloader_num_workers": 4,
}

# LoRA Configuration (for memory efficiency)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FLORENCE_OUTPUT_DIR = OUTPUT_DIR / "florence2"
QWEN_OUTPUT_DIR = OUTPUT_DIR / "qwen2vl"
TEST_DATA_DIR = OUTPUT_DIR / "test_data"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for dir_path in [OUTPUT_DIR, FLORENCE_OUTPUT_DIR, QWEN_OUTPUT_DIR, TEST_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image Processing
IMAGE_SIZE = 384  # Standard size for vision models
MAX_LENGTH = 512  # Max token length for text

# Evaluation Metrics
METRICS = ["mae", "rmse", "r2"]

# Gradio Demo
DEMO_PORT = 7860
DEMO_SHARE = False

# WandB Configuration (optional)
USE_WANDB = False  # Set to True if you want to use Weights & Biases
WANDB_PROJECT = "beaker-volume-detection"
WANDB_ENTITY = None  # Your wandb username

# Filename parsing pattern
# Example: 1284_250ml_v74ml_c_f.jpg
# Format: {id}_{beaker_capacity}ml_v{volume}ml_{background}_{viewpoint}.jpg
FILENAME_PATTERN = r"(\d+)_(\d+)ml_v(\d+)ml_([cu])_([lrabf])\.jpg"

# Background mapping
BACKGROUND_MAP = {
    "c": "controlled",
    "u": "uncontrolled"
}

# Viewpoint mapping
VIEWPOINT_MAP = {
    "l": "left",
    "r": "right",
    "a": "above",
    "b": "below",
    "f": "front"
}

# Prompt templates
FLORENCE_PROMPT_TEMPLATE = "What is the beaker capacity and liquid volume in this image?"
QWEN_PROMPT_TEMPLATE = """You are a precise measurement assistant. Analyze this beaker image and determine:
1. The beaker capacity (100mL or 250mL)
2. The current liquid volume in mL

Respond in the format: "Beaker: {capacity}mL, Volume: {volume}mL"
"""

print(f"Configuration loaded successfully!")
print(f"Dataset: {DATASET_NAME}")
print(f"Output directory: {OUTPUT_DIR}")