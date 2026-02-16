"""
Gradio demo interface for beaker volume detection
"""
import gradio as gr
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration
)
from peft import PeftModel

from config import *
from data_utils import extract_volume_from_text


class BeakerPredictor:
    """Predictor class for inference"""
    
    def __init__(self, model_path: str, model_type: str):
        """
        Args:
            model_path: Path to trained model
            model_type: 'florence' or 'qwen'
        """
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_type} model from {model_path}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        if model_type == "florence":
            base_model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:  # qwen
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        
        self.model.eval()
        print(f"✓ {model_type} model loaded successfully!")
    
    def predict(self, image: Image.Image) -> tuple:
        """
        Predict beaker capacity and liquid volume
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (beaker_capacity, liquid_volume, raw_output)
        """
        if self.model_type == "florence":
            inputs = self.processor(
                text=FLORENCE_PROMPT_TEMPLATE,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False
                )
            
            output_text = self.processor.batch_decode(
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
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=3,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        # Extract volumes
        beaker_capacity, liquid_volume = extract_volume_from_text(output_text)
        
        if beaker_capacity is None:
            beaker_capacity = 0
        if liquid_volume is None:
            liquid_volume = 0
        
        return beaker_capacity, liquid_volume, output_text


def create_demo():
    """Create Gradio demo interface"""
    
    # Check which models are available
    florence_path = FLORENCE_OUTPUT_DIR / "final_model"
    qwen_path = QWEN_OUTPUT_DIR / "final_model"
    
    available_models = []
    if florence_path.exists():
        available_models.append(("Florence-2", str(florence_path), "florence"))
    if qwen_path.exists():
        available_models.append(("Qwen2-VL", str(qwen_path), "qwen"))
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please run train_florence.py and/or train_qwen.py first.")
        return None
    
    # Load models
    predictors = {}
    for model_name, model_path, model_type in available_models:
        try:
            predictors[model_name] = BeakerPredictor(model_path, model_type)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    if not predictors:
        print("❌ Failed to load any models!")
        return None
    
    # Define prediction function
    def predict_volume(image, model_choice):
        """
        Predict volume from image
        
        Args:
            image: Input image (numpy array or PIL)
            model_choice: Selected model name
            
        Returns:
            Formatted prediction string
        """
        if image is None:
            return "Please upload an image."
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Get predictor
        predictor = predictors.get(model_choice)
        if predictor is None:
            return f"Model {model_choice} not available."
        
        try:
            # Predict
            beaker_capacity, liquid_volume, raw_output = predictor.predict(image)
            
            # Format output
            result = f"""
### Prediction Results ({model_choice})

**Beaker Capacity:** {beaker_capacity} mL

**Liquid Volume:** {liquid_volume} mL

---

*Raw Model Output:*
```
{raw_output}
```
            """
            return result
        
        except Exception as e:
            return f"Error during prediction: {str(e)}"
    
    # Create Gradio interface
    with gr.Blocks(title="Beaker Volume Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧪 Beaker Volume Detection
            
            Upload an image of a beaker to predict:
            - **Beaker Capacity** (100mL or 250mL)
            - **Liquid Volume** (in mL)
            
            This model was trained on 5 viewpoints (left, right, above, below, front) 
            with both controlled and uncontrolled backgrounds.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                image_input = gr.Image(
                    label="Upload Beaker Image",
                    type="pil",
                    height=400
                )
                
                model_dropdown = gr.Dropdown(
                    choices=list(predictors.keys()),
                    value=list(predictors.keys())[0],
                    label="Select Model",
                    info="Choose which model to use for prediction"
                )
                
                predict_btn = gr.Button(
                    "🔍 Predict Volume",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                output_text = gr.Markdown(
                    label="Prediction",
                    value="*Upload an image and click 'Predict Volume' to see results.*"
                )
        
        # Example images section
        if TEST_DATA_DIR.exists():
            test_images_dir = TEST_DATA_DIR / "images"
            if test_images_dir.exists():
                example_images = list(test_images_dir.glob("*.jpg"))[:5]
                if example_images:
                    gr.Markdown("### 📸 Example Images")
                    gr.Examples(
                        examples=[[str(img), list(predictors.keys())[0]] for img in example_images],
                        inputs=[image_input, model_dropdown],
                        outputs=output_text,
                        fn=predict_volume,
                        cache_examples=False
                    )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Model Information:**
            - Training Dataset: 2000+ beaker images from Hugging Face
            - Split: 70% train, 15% validation, 15% test
            - Models: Florence-2 and Qwen2-VL (fine-tuned with LoRA)
            - Evaluation Metrics: MAE, RMSE, R²
            
            *For more details, check the evaluation results in the outputs folder.*
            """
        )
        
        # Connect button to prediction function
        predict_btn.click(
            fn=predict_volume,
            inputs=[image_input, model_dropdown],
            outputs=output_text
        )
    
    return demo


def main():
    """Launch Gradio demo"""
    print("=" * 60)
    print("BEAKER VOLUME DETECTION - GRADIO DEMO")
    print("=" * 60)
    
    # Create demo
    demo = create_demo()
    
    if demo is None:
        return
    
    # Launch
    print("\nLaunching Gradio interface...")
    print(f"Model outputs directory: {OUTPUT_DIR}")
    print(f"Test data directory: {TEST_DATA_DIR}")
    
    demo.launch(
        server_port=DEMO_PORT,
        share=DEMO_SHARE,
        server_name="0.0.0.0"  # Accessible from any IP
    )


if __name__ == "__main__":
    main()