"""
Training script for Qwen2-VL model on beaker volume detection
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from config import *
from data_utils import load_and_split_dataset, BeakerDataset, save_test_data

# Set random seeds
torch.manual_seed(DATASET_SPLIT_SEED)
np.random.seed(DATASET_SPLIT_SEED)


class Qwen2VLTrainer:
    """Trainer class for Qwen2-VL model"""
    
    def __init__(self, output_dir: Path = QWEN_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Qwen2-VL Beaker Volume Detection Training")
        print("=" * 60)
        
        # Load processor and model
        print(f"\nLoading Qwen2-VL model: {QWEN_MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16 if QWEN_CONFIG['fp16'] else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Load dataset
        print("\nLoading and splitting dataset...")
        self.dataset_dict = load_and_split_dataset()
        
        # Create datasets
        print("\nPreparing datasets...")
        self.train_dataset = BeakerDataset(
            self.dataset_dict['train'],
            self.processor,
            QWEN_PROMPT_TEMPLATE,
            model_type="qwen"
        )
        
        self.val_dataset = BeakerDataset(
            self.dataset_dict['validation'],
            self.processor,
            QWEN_PROMPT_TEMPLATE,
            model_type="qwen"
        )
        
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
    
    def collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        # Handle pixel values (may be empty for text-only)
        pixel_values = [item['pixel_values'] for item in batch if item['pixel_values'].numel() > 0]
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
        
        # Handle image grid
        image_grid_thw = [item['image_grid_thw'] for item in batch if item['image_grid_thw'].numel() > 0]
        if image_grid_thw:
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            image_grid_thw = None
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if pixel_values is not None:
            result['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            result['image_grid_thw'] = image_grid_thw
        
        return result
    
    def train(self):
        """Train the model"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=QWEN_CONFIG['num_epochs'],
            per_device_train_batch_size=QWEN_CONFIG['batch_size'],
            per_device_eval_batch_size=QWEN_CONFIG['batch_size'],
            gradient_accumulation_steps=QWEN_CONFIG['gradient_accumulation_steps'],
            learning_rate=QWEN_CONFIG['learning_rate'],
            weight_decay=QWEN_CONFIG['weight_decay'],
            warmup_steps=QWEN_CONFIG['warmup_steps'],
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=QWEN_CONFIG['logging_steps'],
            eval_strategy="steps",
            eval_steps=QWEN_CONFIG['eval_steps'],
            save_strategy="steps",
            save_steps=QWEN_CONFIG['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=QWEN_CONFIG['fp16'],
            dataloader_num_workers=QWEN_CONFIG['dataloader_num_workers'],
            remove_unused_columns=False,
            report_to="tensorboard",
            push_to_hub=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save the final model
        print("\nSaving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.processor.save_pretrained(str(self.output_dir / "final_model"))
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Save training summary
        summary = {
            "model": QWEN_MODEL_NAME,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "num_epochs": QWEN_CONFIG['num_epochs'],
            "batch_size": QWEN_CONFIG['batch_size'],
            "learning_rate": QWEN_CONFIG['learning_rate'],
            "final_train_loss": metrics.get('train_loss', None),
        }
        
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {self.output_dir / 'final_model'}")
        print("=" * 60)
        
        return trainer


def main():
    """Main training function"""
    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize trainer
    trainer = Qwen2VLTrainer()
    
    # Train the model
    trainer.train()
    
    print("\n✓ Qwen2-VL training pipeline completed!")
    print(f"  Model checkpoint: {QWEN_OUTPUT_DIR / 'final_model'}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to evaluate the model")
    print("  2. Run demo.py to launch Gradio interface")


if __name__ == "__main__":
    main()