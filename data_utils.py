"""
Data utilities for loading and processing the beaker dataset.
Uses LAZY loading - images are NOT loaded into RAM upfront.
Only metadata is loaded during split processing; images are
fetched on-demand inside __getitem__.
"""

import re
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    Does NOT touch the image - only parses the string.
    """
    name = filename.lower().replace('.jpg','').replace('.jpeg','').replace('.png','')
    parts = name.split('_')

    if len(parts) < 3:
        return None

    try:
        image_id = parts[0]

        # Beaker capacity  e.g. "100ml" or "250ml"
        cap_match = re.match(r'^(\d+)ml$', parts[1])
        if not cap_match:
            return None
        beaker_capacity = int(cap_match.group(1))

        # Liquid volume  e.g. "v74ml", "84ml", "33.5ml", "v29.7ml"
        vol_match = re.match(r'^v?(\d+\.?\d*)ml$', parts[2])
        if not vol_match:
            return None
        liquid_volume = float(vol_match.group(1))

        # Background and viewpoint from remaining parts
        background = 'unknown'
        viewpoint  = 'unknown'
        remaining  = parts[3:]

        def clean_view(token: str) -> str:
            t = token.rstrip('0123456789')
            if len(t) > 1 and len(set(t)) == 1:
                t = t[0]
            return t[0] if t else ''

        if len(remaining) == 1:
            token = clean_view(remaining[0])
            if token in VALID_VIEWS:
                viewpoint = token
            elif token in VALID_BGS:
                background = token

        elif len(remaining) == 2:
            bg_raw   = remaining[0]
            view_raw = clean_view(remaining[1])
            if bg_raw in VALID_BGS:
                background = bg_raw
            if view_raw in VALID_VIEWS:
                viewpoint = view_raw

        elif len(remaining) > 2:
            bg_raw   = remaining[0]
            view_raw = clean_view(remaining[-1])
            if bg_raw in VALID_BGS:
                background = bg_raw
            if view_raw in VALID_VIEWS:
                viewpoint = view_raw

        return {
            'id':              image_id,
            'beaker_capacity': beaker_capacity,
            'liquid_volume':   liquid_volume,
            'background':      BG_MAP.get(background, background),
            'viewpoint':       VIEW_MAP.get(viewpoint, viewpoint),
        }

    except Exception:
        return None


# ── Dataset loading — METADATA ONLY, no image loading ───────────────────────

def load_and_split_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load dataset from HuggingFace.

    KEY CHANGE: We set keep_in_memory=False and do NOT iterate the full
    dataset to collect images. We only extract lightweight metadata
    (filename, label values, index) so that images are fetched lazily
    inside BeakerDataset.__getitem__ via dataset[idx]['image'].

    Returns DatasetDict with keys 'train', 'validation', 'test'.
    """
    print(f"Loading dataset: {DATASET_NAME}")
    print("(Using lazy image loading - metadata only pass)")

    raw = load_dataset(
        DATASET_NAME,
        cache_dir=cache_dir,
        keep_in_memory=False   # <-- critical: do NOT load all images into RAM
    )

    splits_out  = {}
    total_skip  = 0

    for split_name in ['train', 'validation', 'test']:
        if split_name not in raw:
            print(f"  Warning: split '{split_name}' not found, skipping.")
            continue

        split_ds = raw[split_name]
        n        = len(split_ds)
        print(f"\nIndexing split '{split_name}' ({n} samples) — filenames only ...")

        meta_rows = []
        skipped   = 0

        # We only read the filename column here — no image decoding
        # HuggingFace datasets support column-level access
        try:
            filenames = split_ds['image_name']
        except KeyError:
            try:
                filenames = split_ds['file_name']
            except KeyError:
                filenames = [None] * n

        for idx, filename in enumerate(filenames):
            if not filename:
                skipped += 1
                continue

            meta = parse_filename(filename)
            if meta is None:
                skipped += 1
                continue

            meta_rows.append({
                'split_index':     idx,        # index into the raw HF split
                'image_name':      filename,
                'image_id':        meta['id'],
                'beaker_capacity': meta['beaker_capacity'],
                'liquid_volume':   meta['liquid_volume'],
                'background':      meta['background'],
                'viewpoint':       meta['viewpoint'],
            })

        total_skip += skipped
        print(f"  Indexed : {len(meta_rows)}")
        print(f"  Skipped : {skipped} (unparseable filenames)")

        # Store the raw HF split alongside metadata so __getitem__ can
        # do: raw_split[meta_row['split_index']]['image']
        splits_out[split_name] = {
            'meta':      meta_rows,
            'raw_split': split_ds,
        }

    # Fall back to manual split if HF repo has no pre-made splits
    if len(splits_out) < 2:
        print("\nFewer than 2 splits — performing manual 70/15/15 split ...")
        key      = list(splits_out.keys())[0]
        all_meta = splits_out[key]['meta']
        raw_sp   = splits_out[key]['raw_split']
        indices  = list(range(len(all_meta)))

        tv_idx, test_idx = train_test_split(
            indices, test_size=TEST_RATIO,
            random_state=DATASET_SPLIT_SEED, shuffle=True
        )
        train_idx, val_idx = train_test_split(
            tv_idx,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=DATASET_SPLIT_SEED, shuffle=True
        )

        splits_out = {
            'train':      {'meta': [all_meta[i] for i in train_idx], 'raw_split': raw_sp},
            'validation': {'meta': [all_meta[i] for i in val_idx],   'raw_split': raw_sp},
            'test':       {'meta': [all_meta[i] for i in test_idx],  'raw_split': raw_sp},
        }

    total = sum(len(v['meta']) for v in splits_out.values())
    print(f"\n{'='*50}")
    print(f"Dataset ready  —  {total} usable samples")
    for k, v in splits_out.items():
        print(f"  {k:<12}: {len(v['meta']):>5} samples")
    print(f"  ({total_skip} samples skipped — unparseable filenames)")
    print(f"{'='*50}")

    return splits_out          # dict of {'train': {...}, 'validation': {...}, 'test': {...}}


# ── Save test data (lazy — save one image at a time) ─────────────────────────

def save_test_data(splits: dict, output_dir: Path = TEST_DATA_DIR) -> Path:
    """
    Save test-split images + metadata JSON to a folder for demo use.
    Images are loaded and saved one at a time to avoid RAM spikes.
    """
    test_info  = splits['test']
    meta_rows  = test_info['meta']
    raw_split  = test_info['raw_split']

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(meta_rows)} test images to {output_dir} ...")

    metadata = []
    for row in meta_rows:
        # Load only this one image on demand
        sample = raw_split[row['split_index']]
        img    = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        else:
            img = img.convert('RGB')

        fname = row['image_name']
        img.save(images_dir / fname)

        metadata.append({
            'image_name':      fname,
            'image_id':        row['image_id'],
            'beaker_capacity': row['beaker_capacity'],
            'liquid_volume':   row['liquid_volume'],
            'background':      row['background'],
            'viewpoint':       row['viewpoint'],
        })

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "README.txt", 'w') as f:
        f.write("Beaker Volume Detection – Test Dataset\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total samples : {len(meta_rows)}\n")
        f.write(f"Images folder : images/\n")
        f.write(f"Metadata file : metadata.json\n\n")
        f.write("Use these images for the Gradio demo.\n")

    print(f"  Saved {len(meta_rows)} images + metadata.json + README.txt")
    return output_dir


# ── Volume extraction from model text output ──────────────────────────────────

def extract_volume_from_text(text: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Extract beaker capacity and liquid volume from model output text.
    Expected: "Beaker: 250mL, Volume: 74mL"
    """
    text = text.strip()
    b = re.search(r'beaker\s*[:\-]?\s*(\d+)\s*ml',      text, re.IGNORECASE)
    v = re.search(r'volume\s*[:\-]?\s*(\d+\.?\d*)\s*ml', text, re.IGNORECASE)
    return (int(b.group(1)) if b else None,
            float(v.group(1)) if v else None)


# ── PyTorch Dataset — lazy image loading ─────────────────────────────────────

class BeakerDataset(Dataset):
    """
    PyTorch Dataset for beaker volume detection.
    Images are loaded lazily inside __getitem__ — no upfront RAM spike.
    """

    def __init__(self, split_info: dict, processor,
                 prompt_template: str, model_type: str = "florence"):
        """
        Args:
            split_info    : one value from load_and_split_dataset()
                            e.g. splits['train']
                            must have keys 'meta' and 'raw_split'
            processor     : HuggingFace processor for the model
            prompt_template: text prompt
            model_type    : 'florence' or 'qwen'
        """
        self.meta       = split_info['meta']
        self.raw_split  = split_info['raw_split']
        self.processor  = processor
        self.prompt     = prompt_template
        self.model_type = model_type

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta[idx]

        # ── Load image lazily (only this one image, right now) ────────────
        sample = self.raw_split[row['split_index']]
        image  = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        beaker_cap  = row['beaker_capacity']
        liquid_vol  = row['liquid_volume']
        target_text = f"Beaker: {int(beaker_cap)}mL, Volume: {liquid_vol}mL"

        # ── Florence-2 ────────────────────────────────────────────────────
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
                'pixel_values':    inputs['pixel_values'].squeeze(0),
                'input_ids':       inputs['input_ids'].squeeze(0),
                'attention_mask':  inputs.get(
                    'attention_mask',
                    torch.ones_like(inputs['input_ids'])
                ).squeeze(0),
                'labels':          targets['input_ids'].squeeze(0),
                'beaker_capacity': beaker_cap,
                'liquid_volume':   liquid_vol,
                'image_name':      row['image_name'],
            }

        # ── Qwen2-VL ──────────────────────────────────────────────────────
        else:
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

            text_input   = self.processor.apply_chat_template(
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
                'input_ids':       inputs['input_ids'].squeeze(0),
                'attention_mask':  inputs['attention_mask'].squeeze(0),
                'pixel_values':    inputs.get('pixel_values',   torch.tensor([])),
                'image_grid_thw':  inputs.get('image_grid_thw', torch.tensor([])),
                'labels':          target_ids.squeeze(0),
                'beaker_capacity': beaker_cap,
                'liquid_volume':   liquid_vol,
                'image_name':      row['image_name'],
            }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data loading (lazy mode)...\n")

    splits = load_and_split_dataset()

    print("\nSample from train split (metadata only — no image loaded yet):")
    row = splits['train']['meta'][0]
    print(f"  image_name      : {row['image_name']}")
    print(f"  beaker_capacity : {row['beaker_capacity']} mL")
    print(f"  liquid_volume   : {row['liquid_volume']} mL")
    print(f"  background      : {row['background']}")
    print(f"  viewpoint       : {row['viewpoint']}")

    print("\nLoading one image on-demand to verify lazy access ...")
    sample = splits['train']['raw_split'][row['split_index']]
    img    = sample['image']
    print(f"  Image loaded: {img.size if hasattr(img, 'size') else type(img)}")

    save_test_data(splits)

    print("\n✓ data_utils.py completed successfully!")
    print(f"  Train      : {len(splits['train']['meta'])}")
    print(f"  Validation : {len(splits['validation']['meta'])}")
    print(f"  Test       : {len(splits['test']['meta'])}")