"""
Standalone inference script for single image prediction
"""
import argparse
from pathlib import Path
from PIL import Image
import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration
)
from peft import PeftModel

from config import *
from data_utils import extract_volume_from_text


def load_model(model_path: str, model_type: str):
    """Load model for inference"""
    print(f"Loading {model_type} model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if model_type == "florence":
        base_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    print("✓ Model loaded successfully!")
    
    return processor, model


def predict_image(image_path: str, model_type: str, model_path: str = None):
    """
    Predict volume from a single image
    
    Args:
        image_path: Path to image file
        model_type: 'florence' or 'qwen'
        model_path: Optional custom model path
    """
    # Determine model path
    if model_path is None:
        if model_type == "florence":
            model_path = FLORENCE_OUTPUT_DIR / "final_model"
        else:
            model_path = QWEN_OUTPUT_DIR / "final_model"
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"Please train the {model_type} model first!")
        return
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"\n📸 Loaded image: {image_path}")
    print(f"   Size: {image.size}")
    
    # Load model
    processor, model = load_model(str(model_path), model_type)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Predict
    print("\n🔍 Predicting...")
    
    if model_type == "florence":
        inputs = processor(
            text=FLORENCE_PROMPT_TEMPLATE,
            images=image,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False
            )
        
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
    
    else:  # qwen
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QWEN_PROMPT_TEMPLATE}
                ]
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    # Extract results
    beaker_capacity, liquid_volume = extract_volume_from_text(output_text)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Model: {model_type.upper()}")
    print(f"\nBeaker Capacity: {beaker_capacity} mL")
    print(f"Liquid Volume:   {liquid_volume} mL")
    print(f"\nRaw Output: {output_text}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Beaker Volume Detection - Inference")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to beaker image"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["florence", "qwen"],
        default="florence",
        help="Model to use for prediction (default: florence)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to model checkpoint (optional)"
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict_image(args.image_path, args.model, args.model_path)


if __name__ == "__main__":
    main()