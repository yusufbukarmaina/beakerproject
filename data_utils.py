"""
Data utilities for loading and processing the beaker dataset
"""
import re
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from transformers import AutoProcessor
from sklearn.model_selection import train_test_split

from config import *


@dataclass
class BeakerSample:
    """Data class for a single beaker sample"""
    image_id: str
    image_path: str
    beaker_capacity: int  # 100 or 250
    liquid_volume: int    # actual volume in mL
    background: str       # controlled or uncontrolled
    viewpoint: str        # left, right, above, below, front
    image: Optional[Image.Image] = None


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse beaker image filename to extract metadata
    
    Example: 1284_250ml_v74ml_c_f.jpg
    Returns: {
        'id': '1284',
        'beaker_capacity': 250,
        'liquid_volume': 74,
        'background': 'controlled',
        'viewpoint': 'front'
    }
    """
    match = re.match(FILENAME_PATTERN, filename)
    if not match:
        return None
    
    image_id, capacity, volume, bg, view = match.groups()
    
    return {
        'id': image_id,
        'beaker_capacity': int(capacity),
        'liquid_volume': int(volume),
        'background': BACKGROUND_MAP.get(bg, bg),
        'viewpoint': VIEWPOINT_MAP.get(view, view)
    }


def load_and_split_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load dataset from Hugging Face and split into train/val/test
    
    Args:
        cache_dir: Directory to cache the dataset
        
    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits
    """
    print(f"Loading dataset: {DATASET_NAME}")
    
    # Load the dataset
    dataset = load_dataset(DATASET_NAME, cache_dir=cache_dir)
    
    # Get the train split (assuming dataset has 'train' split)
    if 'train' in dataset:
        full_dataset = dataset['train']
    else:
        # If no split exists, take the first available split
        full_dataset = dataset[list(dataset.keys())[0]]
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Parse filenames to extract labels
    parsed_data = []
    for idx, sample in enumerate(full_dataset):
        filename = sample.get('image_name', '') or sample.get('file_name', '')
        
        if not filename:
            print(f"Warning: No filename found for sample {idx}")
            continue
            
        metadata = parse_filename(filename)
        if metadata:
            parsed_data.append({
                'image': sample['image'],
                'image_name': filename,
                'image_id': metadata['id'],
                'beaker_capacity': metadata['beaker_capacity'],
                'liquid_volume': metadata['liquid_volume'],
                'background': metadata['background'],
                'viewpoint': metadata['viewpoint']
            })
        else:
            print(f"Warning: Could not parse filename: {filename}")
    
    print(f"Successfully parsed {len(parsed_data)} samples")
    
    # Create indices for splitting
    indices = list(range(len(parsed_data)))
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_RATIO,
        random_state=DATASET_SPLIT_SEED,
        shuffle=True
    )
    
    # Second split: separate train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=DATASET_SPLIT_SEED,
        shuffle=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/len(indices)*100:.1f}%)")
    print(f"  Validation: {len(val_indices)} samples ({len(val_indices)/len(indices)*100:.1f}%)")
    print(f"  Test: {len(test_indices)} samples ({len(test_indices)/len(indices)*100:.1f}%)")
    
    # Create split datasets
    train_data = [parsed_data[i] for i in train_indices]
    val_data = [parsed_data[i] for i in val_indices]
    test_data = [parsed_data[i] for i in test_indices]
    
    # Convert to HuggingFace datasets
    from datasets import Dataset as HFDataset
    
    dataset_dict = DatasetDict({
        'train': HFDataset.from_list(train_data),
        'validation': HFDataset.from_list(val_data),
        'test': HFDataset.from_list(test_data)
    })
    
    return dataset_dict


def save_test_data(dataset_dict: DatasetDict, output_dir: Path = TEST_DATA_DIR):
    """
    Save test dataset images and metadata to a folder for demo purposes
    
    Args:
        dataset_dict: Dataset dictionary containing test split
        output_dir: Directory to save test data
    """
    test_dataset = dataset_dict['test']
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving test data to {output_dir}")
    
    # Save images and create metadata
    metadata = []
    for idx, sample in enumerate(test_dataset):
        image = sample['image']
        filename = sample['image_name']
        
        # Save image
        image_path = images_dir / filename
        image.save(image_path)
        
        # Store metadata
        metadata.append({
            'image_name': filename,
            'image_id': sample['image_id'],
            'beaker_capacity': sample['beaker_capacity'],
            'liquid_volume': sample['liquid_volume'],
            'background': sample['background'],
            'viewpoint': sample['viewpoint']
        })
    
    # Save metadata as JSON
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create a README
    readme_path = output_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Beaker Volume Detection - Test Dataset\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total test samples: {len(test_dataset)}\n")
        f.write(f"Images directory: {images_dir}\n")
        f.write(f"Metadata file: {metadata_path}\n\n")
        f.write(f"Use these images for model evaluation and Gradio demo.\n")
    
    print(f"✓ Saved {len(test_dataset)} test images")
    print(f"✓ Test data ready at: {output_dir}")
    
    return output_dir


class BeakerDataset(Dataset):
    """PyTorch Dataset for beaker volume detection"""
    
    def __init__(
        self,
        hf_dataset: HFDataset,
        processor,
        prompt_template: str,
        model_type: str = "florence"
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset
            processor: Model processor/tokenizer
            prompt_template: Prompt template for the model
            model_type: 'florence' or 'qwen'
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.prompt_template = prompt_template
        self.model_type = model_type
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        
        # Create target text
        beaker_capacity = sample['beaker_capacity']
        liquid_volume = sample['liquid_volume']
        target_text = f"Beaker: {beaker_capacity}mL, Volume: {liquid_volume}mL"
        
        if self.model_type == "florence":
            # Florence-2 processing
            inputs = self.processor(
                text=self.prompt_template,
                images=image,
                return_tensors="pt"
            )
            
            # Process target
            targets = self.processor.tokenizer(
                text=target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True
            )
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])).squeeze(0),
                'labels': targets['input_ids'].squeeze(0),
                'beaker_capacity': beaker_capacity,
                'liquid_volume': liquid_volume,
                'image_name': sample['image_name']
            }
        
        else:  # qwen
            # Qwen2-VL processing with conversation format
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt_template}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Process target
            target_ids = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True
            )['input_ids']
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs.get('pixel_values', torch.tensor([])),
                'image_grid_thw': inputs.get('image_grid_thw', torch.tensor([])),
                'labels': target_ids.squeeze(0),
                'beaker_capacity': beaker_capacity,
                'liquid_volume': liquid_volume,
                'image_name': sample['image_name']
            }


def extract_volume_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract beaker capacity and liquid volume from model output
    
    Args:
        text: Model output text
        
    Returns:
        Tuple of (beaker_capacity, liquid_volume) or (None, None) if parsing fails
    """
    # Expected format: "Beaker: 250mL, Volume: 74mL"
    beaker_pattern = r"Beaker:\s*(\d+)\s*mL"
    volume_pattern = r"Volume:\s*(\d+)\s*mL"
    
    beaker_match = re.search(beaker_pattern, text, re.IGNORECASE)
    volume_match = re.search(volume_pattern, text, re.IGNORECASE)
    
    beaker_capacity = int(beaker_match.group(1)) if beaker_match else None
    liquid_volume = int(volume_match.group(1)) if volume_match else None
    
    return beaker_capacity, liquid_volume


if __name__ == "__main__":
    # Test the data loading
    print("Testing data loading...")
    dataset_dict = load_and_split_dataset()
    
    print("\nSample from train set:")
    sample = dataset_dict['train'][0]
    print(f"  Image name: {sample['image_name']}")
    print(f"  Beaker capacity: {sample['beaker_capacity']}mL")
    print(f"  Liquid volume: {sample['liquid_volume']}mL")
    print(f"  Background: {sample['background']}")
    print(f"  Viewpoint: {sample['viewpoint']}")
    
    # Save test data
    save_test_data(dataset_dict)
    print("\n✓ Data utilities test completed successfully!")