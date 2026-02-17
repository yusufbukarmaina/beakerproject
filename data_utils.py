"""
Data utilities for loading and processing the beaker dataset.
Handles ALL filename variations observed in yusufbukarmaina/Beakers1:
  - Mixed case mL/ml
  - Optional 'v' prefix on volume  (v74ml  or  84mL)
  - Decimal volumes               (33.5ml, 29.7ml, 107.5mL)
  - Missing background field      (653_100ml_33.5mL_b.jpg)
  - Doubled viewpoint chars       (ff, aa)
  - Viewpoint with trailing digit (r1, l2)
  - Uppercase viewpoint           (L -> l)
  - Dataset already has train/val/test splits on HuggingFace
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
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split

from config import (
    DATASET_NAME, DATASET_SPLIT_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TEST_DATA_DIR, MAX_LENGTH,
    FLORENCE_PROMPT_TEMPLATE, QWEN_PROMPT_TEMPLATE
)

# ── Constants ────────────────────────────────────────────────────────────────

VALID_VIEWS = {'l', 'r', 'a', 'b', 'f'}
VALID_BGS   = {'c', 'u'}

BG_MAP = {
    'c':       'controlled',
    'u':       'uncontrolled',
    'unknown': 'unknown'
}

VIEW_MAP = {
    'l':       'left',
    'r':       'right',
    'a':       'above',
    'b':       'below',
    'f':       'front',
    'unknown': 'unknown'
}


# ── Filename Parser ──────────────────────────────────────────────────────────

def parse_filename(filename: str) -> Optional[Dict]:
    """
    Robust filename parser that handles all observed variations.

    Examples handled:
        1284_250ml_v74ml_c_f.jpg      -> standard format
        1859_100mL_84mL_u_a.jpg       -> no 'v', mixed case
        2524_100mL_33.5ml_u_l.jpg     -> decimal volume, no 'v'
        2088_250mL_v29.7ml_u_r.jpg    -> decimal volume with 'v'
        653_100ml_33.5mL_b.jpg        -> missing background
        1528_100mL_19mL_r.jpg         -> missing background
        2119_250mL_v37ml_u_ff.jpg     -> doubled viewpoint char
        994_250ml_v2ml_c_L.jpg        -> uppercase viewpoint
        1042_250ml_v12ml_c_r1.jpg     -> viewpoint with digit suffix
        1947_250mL_107.5mL_u_l.jpg    -> large decimal volume

    Returns dict with keys: id, beaker_capacity, liquid_volume,
                             background, viewpoint
    Returns None if parsing fails.
    """
    name = filename.lower().replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    parts = name.split('_')

    if len(parts) < 3:
        return None

    try:
        image_id = parts[0]

        # ── Beaker capacity ────────────────────────────────────────────────
        # e.g. "100ml" or "250ml"
        cap_match = re.match(r'^(\d+)ml$', parts[1])
        if not cap_match:
            return None
        beaker_capacity = int(cap_match.group(1))

        # ── Liquid volume ──────────────────────────────────────────────────
        # e.g. "v74ml", "84ml", "33.5ml", "v29.7ml", "107.5ml"
        vol_match = re.match(r'^v?(\d+\.?\d*)ml$', parts[2])
        if not vol_match:
            return None
        liquid_volume = float(vol_match.group(1))

        # ── Background and viewpoint ───────────────────────────────────────
        # Remaining parts after index 2
        background = 'unknown'
        viewpoint  = 'unknown'
        remaining  = parts[3:]

        def clean_viewpoint(token: str) -> str:
            """Strip trailing digits, collapse doubled chars, take first char."""
            t = token.rstrip('0123456789')          # remove r1 -> r
            if len(t) > 1 and len(set(t)) == 1:
                t = t[0]                             # ff -> f, aa -> a
            return t[0] if t else ''

        if len(remaining) == 0:
            pass  # no bg/view info

        elif len(remaining) == 1:
            # Either just a viewpoint or just a background
            token = clean_viewpoint(remaining[0])
            if token in VALID_VIEWS:
                viewpoint = token
            elif token in VALID_BGS:
                background = token

        elif len(remaining) == 2:
            # Standard: [bg, view]
            bg_raw   = remaining[0]
            view_raw = clean_viewpoint(remaining[1])
            if bg_raw in VALID_BGS:
                background = bg_raw
            if view_raw in VALID_VIEWS:
                viewpoint = view_raw

        else:
            # More than 2 remaining tokens - take first as bg, last as view
            bg_raw   = remaining[0]
            view_raw = clean_viewpoint(remaining[-1])
            if bg_raw in VALID_BGS:
                background = bg_raw
            if view_raw in VALID_VIEWS:
                viewpoint = view_raw

        return {
            'id':               image_id,
            'beaker_capacity':  beaker_capacity,
            'liquid_volume':    liquid_volume,
            'background':       BG_MAP.get(background, background),
            'viewpoint':        VIEW_MAP.get(viewpoint, viewpoint)
        }

    except Exception:
        return None


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_and_split_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load dataset from HuggingFace.

    The dataset already has train / validation / test splits, so we
    respect those. We still parse filenames to attach the metadata columns
    (beaker_capacity, liquid_volume, background, viewpoint).

    Returns DatasetDict with keys 'train', 'validation', 'test'.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME, cache_dir=cache_dir)

    splits_out = {}
    total_skipped = 0

    for split_name in ['train', 'validation', 'test']:
        if split_name not in raw:
            print(f"  Warning: split '{split_name}' not found in dataset, skipping.")
            continue

        split_data = raw[split_name]
        print(f"\nProcessing split '{split_name}' ({len(split_data)} samples)...")

        parsed_rows = []
        skipped = 0

        for sample in split_data:
            # Filename may be stored under different column names
            filename = (sample.get('image_name') or
                        sample.get('file_name')  or
                        sample.get('filename')   or '')

            if not filename:
                skipped += 1
                continue

            meta = parse_filename(filename)
            if meta is None:
                skipped += 1
                continue

            parsed_rows.append({
                'image':            sample['image'],
                'image_name':       filename,
                'image_id':         meta['id'],
                'beaker_capacity':  meta['beaker_capacity'],
                'liquid_volume':    meta['liquid_volume'],
                'background':       meta['background'],
                'viewpoint':        meta['viewpoint'],
            })

        total_skipped += skipped
        print(f"  Parsed:  {len(parsed_rows)}")
        print(f"  Skipped: {skipped} (unrecognised filename format)")

        splits_out[split_name] = HFDataset.from_list(parsed_rows)

    # If the HuggingFace repo doesn't have pre-made splits, fall back to
    # manual splitting of the first available split.
    if len(splits_out) < 2:
        print("\nFewer than 2 splits found - performing manual 70/15/15 split...")
        all_key   = list(splits_out.keys())[0]
        all_data  = list(splits_out[all_key])
        indices   = list(range(len(all_data)))

        tv_idx, test_idx = train_test_split(
            indices, test_size=TEST_RATIO, random_state=DATASET_SPLIT_SEED, shuffle=True
        )
        train_idx, val_idx = train_test_split(
            tv_idx,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=DATASET_SPLIT_SEED,
            shuffle=True
        )

        splits_out = {
            'train':      HFDataset.from_list([all_data[i] for i in train_idx]),
            'validation': HFDataset.from_list([all_data[i] for i in val_idx]),
            'test':       HFDataset.from_list([all_data[i] for i in test_idx]),
        }

    dataset_dict = DatasetDict(splits_out)

    total = sum(len(v) for v in splits_out.values())
    print(f"\n{'='*50}")
    print(f"Dataset ready — {total} usable samples")
    for k, v in splits_out.items():
        print(f"  {k:<12}: {len(v):>5} samples")
    print(f"  (skipped {total_skipped} samples with unparseable filenames)")
    print(f"{'='*50}")

    return dataset_dict


# ── Save test data ────────────────────────────────────────────────────────────

def save_test_data(dataset_dict: DatasetDict,
                   output_dir: Path = TEST_DATA_DIR) -> Path:
    """
    Save test-split images + metadata JSON to a folder for demo use.
    Returns the output directory path.
    """
    test_ds = dataset_dict['test']

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(test_ds)} test images to {output_dir} ...")

    metadata = []
    for sample in test_ds:
        img = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')

        fname = sample['image_name']
        img.save(images_dir / fname)

        metadata.append({
            'image_name':      fname,
            'image_id':        sample['image_id'],
            'beaker_capacity': sample['beaker_capacity'],
            'liquid_volume':   sample['liquid_volume'],
            'background':      sample['background'],
            'viewpoint':       sample['viewpoint'],
        })

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "README.txt", 'w') as f:
        f.write("Beaker Volume Detection – Test Dataset\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total samples : {len(test_ds)}\n")
        f.write(f"Images folder : images/\n")
        f.write(f"Metadata      : metadata.json\n\n")
        f.write("Use these images for the Gradio demo.\n")

    print(f"  Saved {len(test_ds)} images  +  metadata.json  +  README.txt")
    return output_dir


# ── Volume extraction from model text output ──────────────────────────────────

def extract_volume_from_text(text: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Extract beaker capacity and liquid volume from a model's text output.

    Expected format: "Beaker: 250mL, Volume: 74mL"
    Also handles:    "beaker: 250ml volume: 74ml" (case insensitive)
    """
    text = text.strip()

    beaker_pat = r'beaker\s*[:\-]?\s*(\d+)\s*ml'
    volume_pat = r'volume\s*[:\-]?\s*(\d+\.?\d*)\s*ml'

    b_match = re.search(beaker_pat, text, re.IGNORECASE)
    v_match = re.search(volume_pat, text, re.IGNORECASE)

    beaker_capacity = int(b_match.group(1))   if b_match else None
    liquid_volume   = float(v_match.group(1)) if v_match else None

    return beaker_capacity, liquid_volume


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class BeakerDataset(Dataset):
    """PyTorch Dataset wrapper for beaker volume detection."""

    def __init__(self, hf_dataset, processor, prompt_template: str,
                 model_type: str = "florence"):
        self.dataset       = hf_dataset
        self.processor     = processor
        self.prompt        = prompt_template
        self.model_type    = model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        image = image.convert('RGB')

        beaker_cap  = sample['beaker_capacity']
        liquid_vol  = sample['liquid_volume']
        target_text = f"Beaker: {int(beaker_cap)}mL, Volume: {liquid_vol}mL"

        if self.model_type == "florence":
            inputs = self.processor(
                text=self.prompt,
                images=image,
                return_tensors="pt"
            )
            targets = self.processor.tokenizer(
                text=target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True
            )
            return {
                'pixel_values':   inputs['pixel_values'].squeeze(0),
                'input_ids':      inputs['input_ids'].squeeze(0),
                'attention_mask': inputs.get(
                    'attention_mask',
                    torch.ones_like(inputs['input_ids'])
                ).squeeze(0),
                'labels':         targets['input_ids'].squeeze(0),
                'beaker_capacity': beaker_cap,
                'liquid_volume':   liquid_vol,
                'image_name':      sample['image_name'],
            }

        else:  # qwen
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  self.prompt}
                    ]
                }
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            target_ids = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True
            )['input_ids']

            return {
                'input_ids':        inputs['input_ids'].squeeze(0),
                'attention_mask':   inputs['attention_mask'].squeeze(0),
                'pixel_values':     inputs.get('pixel_values',    torch.tensor([])),
                'image_grid_thw':   inputs.get('image_grid_thw',  torch.tensor([])),
                'labels':           target_ids.squeeze(0),
                'beaker_capacity':  beaker_cap,
                'liquid_volume':    liquid_vol,
                'image_name':       sample['image_name'],
            }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data loading...\n")

    dataset_dict = load_and_split_dataset()

    print("\nSample from train split:")
    s = dataset_dict['train'][0]
    print(f"  image_name      : {s['image_name']}")
    print(f"  beaker_capacity : {s['beaker_capacity']} mL")
    print(f"  liquid_volume   : {s['liquid_volume']} mL")
    print(f"  background      : {s['background']}")
    print(f"  viewpoint       : {s['viewpoint']}")

    save_test_data(dataset_dict)

    print("\n✓ data_utils.py completed successfully!")
    print(f"  Train : {len(dataset_dict['train'])}")
    print(f"  Val   : {len(dataset_dict['validation'])}")
    print(f"  Test  : {len(dataset_dict['test'])}")