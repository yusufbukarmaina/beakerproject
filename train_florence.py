"""
Training script for Florence-2 model on beaker volume detection
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

from config import *
from data_utils import load_and_split_dataset, BeakerDataset, save_test_data

# Set random seeds for reproducibility
torch.manual_seed(DATASET_SPLIT_SEED)
np.random.seed(DATASET_SPLIT_SEED)


class Florence2Trainer:
    """Trainer class for Florence-2 model"""
    
    def __init__(self, output_dir: Path = FLORENCE_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Florence-2 Beaker Volume Detection Training")
        print("=" * 60)
        
        # Load processor and model
        print(f"\nLoading Florence-2 model: {FLORENCE_MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_NAME,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_NAME,
            torch_dtype=torch.float16 if FLORENCE_CONFIG['fp16'] else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Apply LoRA for efficient fine-tuning
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Load dataset
        print("\nLoading and splitting dataset...")
        self.dataset_dict = load_and_split_dataset()
        
        # Save test data for later use
        print("\nSaving test data...")
        save_test_data(self.dataset_dict)
        
        # Create datasets
        print("\nPreparing datasets...")
        self.train_dataset = BeakerDataset(
            self.dataset_dict['train'],
            self.processor,
            FLORENCE_PROMPT_TEMPLATE,
            model_type="florence"
        )
        
        self.val_dataset = BeakerDataset(
            self.dataset_dict['validation'],
            self.processor,
            FLORENCE_PROMPT_TEMPLATE,
            model_type="florence"
        )
        
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
    
    def collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def train(self):
        """Train the model"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=FLORENCE_CONFIG['num_epochs'],
            per_device_train_batch_size=FLORENCE_CONFIG['batch_size'],
            per_device_eval_batch_size=FLORENCE_CONFIG['batch_size'],
            gradient_accumulation_steps=FLORENCE_CONFIG['gradient_accumulation_steps'],
            learning_rate=FLORENCE_CONFIG['learning_rate'],
            weight_decay=FLORENCE_CONFIG['weight_decay'],
            warmup_steps=FLORENCE_CONFIG['warmup_steps'],
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=FLORENCE_CONFIG['logging_steps'],
            eval_strategy="steps",
            eval_steps=FLORENCE_CONFIG['eval_steps'],
            save_strategy="steps",
            save_steps=FLORENCE_CONFIG['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=FLORENCE_CONFIG['fp16'],
            dataloader_num_workers=FLORENCE_CONFIG['dataloader_num_workers'],
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
            "model": FLORENCE_MODEL_NAME,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "num_epochs": FLORENCE_CONFIG['num_epochs'],
            "batch_size": FLORENCE_CONFIG['batch_size'],
            "learning_rate": FLORENCE_CONFIG['learning_rate'],
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
    trainer = Florence2Trainer()
    
    # Train the model
    trainer.train()
    
    print("\n✓ Florence-2 training pipeline completed!")
    print(f"  Model checkpoint: {FLORENCE_OUTPUT_DIR / 'final_model'}")
    print(f"  Test data: {TEST_DATA_DIR}")
    print("\nNext steps:")
    print("  1. Run evaluate.py to evaluate the model")
    print("  2. Run train_qwen.py to train Qwen2-VL model")
    print("  3. Run demo.py to launch Gradio interface")


if __name__ == "__main__":
    main()