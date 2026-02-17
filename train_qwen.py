"""
Training script for Qwen2-VL on beaker volume detection.
Uses lazy image loading - no timeout on dataset processing.
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

from config import *
from data_utils import load_and_split_dataset, BeakerDataset, save_test_data

torch.manual_seed(DATASET_SPLIT_SEED)
np.random.seed(DATASET_SPLIT_SEED)


class Qwen2VLTrainer:

    def __init__(self, output_dir: Path = QWEN_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Qwen2-VL Beaker Volume Detection Training")
        print("=" * 60)

        # ── Load processor ────────────────────────────────────────────────
        print(f"\nLoading processor: {QWEN_MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True
        )

        # ── Load model ────────────────────────────────────────────────────
        print(f"Loading model: {QWEN_MODEL_NAME}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # ── Apply LoRA ────────────────────────────────────────────────────
        print("Applying LoRA ...")
        lora_cfg = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        # ── Load dataset (metadata only — no images in RAM) ───────────────
        print("\nLoading dataset (lazy mode — metadata only) ...")
        self.splits = load_and_split_dataset()

        # Save test images for demo (done once, lazily)
        print("\nSaving test data for demo ...")
        save_test_data(self.splits)

        # ── Build PyTorch datasets ────────────────────────────────────────
        print("\nBuilding PyTorch datasets ...")
        self.train_dataset = BeakerDataset(
            self.splits['train'],
            self.processor,
            QWEN_PROMPT_TEMPLATE,
            model_type="qwen"
        )
        self.val_dataset = BeakerDataset(
            self.splits['validation'],
            self.processor,
            QWEN_PROMPT_TEMPLATE,
            model_type="qwen"
        )
        print(f"  Train      : {len(self.train_dataset)} samples")
        print(f"  Validation : {len(self.val_dataset)} samples")

    # ── Collate ───────────────────────────────────────────────────────────
    def collate_fn(self, batch):
        input_ids      = torch.stack([b['input_ids']      for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels         = torch.stack([b['labels']         for b in batch])

        pv_list = [b['pixel_values'] for b in batch if b['pixel_values'].numel() > 0]
        pixel_values = torch.cat(pv_list, dim=0) if pv_list else None

        grid_list = [b['image_grid_thw'] for b in batch if b['image_grid_thw'].numel() > 0]
        image_grid_thw = torch.cat(grid_list, dim=0) if grid_list else None

        result = {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
        }
        if pixel_values is not None:
            result['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            result['image_grid_thw'] = image_grid_thw

        return result

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self):
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        args = TrainingArguments(
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
            dataloader_pin_memory=False,   # avoids extra RAM pressure
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
        )

        print("\nTraining started ...")
        result = trainer.train()

        print("\nSaving final model ...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.processor.save_pretrained(str(self.output_dir / "final_model"))

        metrics = result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        summary = {
            "model":            QWEN_MODEL_NAME,
            "train_samples":    len(self.train_dataset),
            "val_samples":      len(self.val_dataset),
            "num_epochs":       QWEN_CONFIG['num_epochs'],
            "learning_rate":    QWEN_CONFIG['learning_rate'],
            "final_train_loss": metrics.get('train_loss'),
        }
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {self.output_dir / 'final_model'}")
        print("=" * 60)
        return trainer


def main():
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"RAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")

    trainer = Qwen2VLTrainer()
    trainer.train()

    print("\n✓ Qwen2-VL training complete!")
    print(f"  Checkpoint : {QWEN_OUTPUT_DIR / 'final_model'}")
    print(f"  Test data  : {TEST_DATA_DIR}")
    print("\nNext: python3 evaluate.py")


if __name__ == "__main__":
    main()