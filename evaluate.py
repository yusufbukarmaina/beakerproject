"""
Evaluation script for both Florence-2 and Qwen2-VL models
Computes MAE, RMSE, and R² metrics on test set
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration
)
from peft import PeftModel

from config import *
from data_utils import load_and_split_dataset, extract_volume_from_text

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Evaluator class for beaker volume detection models"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_path: Path to the trained model
            model_type: 'florence' or 'qwen'
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device
        
        print(f"\nLoading {model_type.upper()} model from {model_path}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if model_type == "florence":
            base_model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.model_path)
            )
        else:  # qwen
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.model_path)
            )
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {device}")
    
    def predict(self, image: Image.Image) -> Tuple[int, int]:
        """
        Predict beaker capacity and liquid volume from image
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_beaker_capacity, predicted_liquid_volume)
        """
        if self.model_type == "florence":
            # Florence-2 inference
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
            
            generated_text = self.processor.batch_decode(
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
            
            # Decode only the generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            generated_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        # Extract volumes from text
        beaker_capacity, liquid_volume = extract_volume_from_text(generated_text)
        
        # If extraction failed, return defaults
        if beaker_capacity is None:
            beaker_capacity = 0
        if liquid_volume is None:
            liquid_volume = 0
        
        return beaker_capacity, liquid_volume
    
    def evaluate(self, test_dataset) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: HuggingFace dataset with test samples
            
        Returns:
            Dictionary with evaluation metrics and predictions
        """
        print(f"\nEvaluating on {len(test_dataset)} test samples...")
        
        predictions = []
        ground_truth = []
        details = []
        
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[idx]
            
            # Get image
            image = sample['image']
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert('RGB')
            
            # Ground truth
            true_beaker = sample['beaker_capacity']
            true_volume = sample['liquid_volume']
            
            # Prediction
            try:
                pred_beaker, pred_volume = self.predict(image)
            except Exception as e:
                print(f"\nError predicting sample {idx}: {e}")
                pred_beaker, pred_volume = 0, 0
            
            predictions.append({
                'beaker_capacity': pred_beaker,
                'liquid_volume': pred_volume
            })
            
            ground_truth.append({
                'beaker_capacity': true_beaker,
                'liquid_volume': true_volume
            })
            
            details.append({
                'image_name': sample['image_name'],
                'true_beaker': true_beaker,
                'true_volume': true_volume,
                'pred_beaker': pred_beaker,
                'pred_volume': pred_volume,
                'beaker_error': abs(pred_beaker - true_beaker),
                'volume_error': abs(pred_volume - true_volume),
                'background': sample['background'],
                'viewpoint': sample['viewpoint']
            })
        
        # Calculate metrics for liquid volume (primary task)
        true_volumes = [gt['liquid_volume'] for gt in ground_truth]
        pred_volumes = [p['liquid_volume'] for p in predictions]
        
        mae = mean_absolute_error(true_volumes, pred_volumes)
        rmse = np.sqrt(mean_squared_error(true_volumes, pred_volumes))
        r2 = r2_score(true_volumes, pred_volumes)
        
        # Calculate metrics for beaker capacity (secondary task)
        true_beakers = [gt['beaker_capacity'] for gt in ground_truth]
        pred_beakers = [p['beaker_capacity'] for p in predictions]
        
        beaker_accuracy = np.mean([t == p for t, p in zip(true_beakers, pred_beakers)])
        
        # Compile results
        results = {
            'model_type': self.model_type,
            'model_path': str(self.model_path),
            'num_samples': len(test_dataset),
            'liquid_volume_metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            },
            'beaker_capacity_metrics': {
                'accuracy': float(beaker_accuracy)
            },
            'predictions': details
        }
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results - {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"\nLiquid Volume Prediction:")
        print(f"  MAE:  {mae:.2f} mL")
        print(f"  RMSE: {rmse:.2f} mL")
        print(f"  R²:   {r2:.4f}")
        print(f"\nBeaker Capacity Classification:")
        print(f"  Accuracy: {beaker_accuracy*100:.2f}%")
        print(f"{'='*60}")
        
        return results


def plot_comparison(florence_results: Dict, qwen_results: Dict, output_dir: Path):
    """
    Create comparison plots between Florence-2 and Qwen2-VL
    
    Args:
        florence_results: Results dictionary from Florence-2
        qwen_results: Results dictionary from Qwen2-VL
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['mae', 'rmse', 'r2']
    metric_names = ['MAE (mL)', 'RMSE (mL)', 'R² Score']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        florence_val = florence_results['liquid_volume_metrics'][metric]
        qwen_val = qwen_results['liquid_volume_metrics'][metric]
        
        axes[idx].bar(['Florence-2', 'Qwen2-VL'], [florence_val, qwen_val], 
                     color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[idx].set_ylabel(name, fontsize=12)
        axes[idx].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate([florence_val, qwen_val]):
            axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metrics_comparison.png")
    plt.close()
    
    # 2. Prediction scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (results, model_name, color) in enumerate([
        (florence_results, 'Florence-2', '#3498db'),
        (qwen_results, 'Qwen2-VL', '#e74c3c')
    ]):
        true_vals = [p['true_volume'] for p in results['predictions']]
        pred_vals = [p['pred_volume'] for p in results['predictions']]
        
        axes[idx].scatter(true_vals, pred_vals, alpha=0.5, color=color, s=30)
        
        # Perfect prediction line
        min_val = min(min(true_vals), min(pred_vals))
        max_val = max(max(true_vals), max(pred_vals))
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                      'k--', linewidth=2, label='Perfect Prediction')
        
        axes[idx].set_xlabel('True Volume (mL)', fontsize=12)
        axes[idx].set_ylabel('Predicted Volume (mL)', fontsize=12)
        axes[idx].set_title(f'{model_name} - Predictions vs Ground Truth', 
                           fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        
        # Add R² to plot
        r2 = results['liquid_volume_metrics']['r2']
        axes[idx].text(0.05, 0.95, f'R² = {r2:.4f}', 
                      transform=axes[idx].transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: prediction_scatter.png")
    plt.close()
    
    # 3. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (results, model_name, color) in enumerate([
        (florence_results, 'Florence-2', '#3498db'),
        (qwen_results, 'Qwen2-VL', '#e74c3c')
    ]):
        errors = [p['volume_error'] for p in results['predictions']]
        
        axes[idx].hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Absolute Error (mL)', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'{model_name} - Error Distribution', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add mean error line
        mean_error = np.mean(errors)
        axes[idx].axvline(mean_error, color='red', linestyle='--', 
                         linewidth=2, label=f'Mean Error: {mean_error:.2f} mL')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: error_distribution.png")
    plt.close()
    
    # 4. Performance by viewpoint
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (results, model_name) in enumerate([
        (florence_results, 'Florence-2'),
        (qwen_results, 'Qwen2-VL')
    ]):
        # Group by viewpoint
        viewpoint_errors = {}
        for p in results['predictions']:
            vp = p['viewpoint']
            if vp not in viewpoint_errors:
                viewpoint_errors[vp] = []
            viewpoint_errors[vp].append(p['volume_error'])
        
        viewpoints = list(viewpoint_errors.keys())
        mean_errors = [np.mean(viewpoint_errors[vp]) for vp in viewpoints]
        
        axes[idx].bar(viewpoints, mean_errors, color='skyblue', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Viewpoint', fontsize=12)
        axes[idx].set_ylabel('Mean Absolute Error (mL)', fontsize=12)
        axes[idx].set_title(f'{model_name} - Performance by Viewpoint', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(mean_errors):
            axes[idx].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_viewpoint.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_by_viewpoint.png")
    plt.close()
    
    # 5. Performance by background
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (results, model_name) in enumerate([
        (florence_results, 'Florence-2'),
        (qwen_results, 'Qwen2-VL')
    ]):
        # Group by background
        bg_errors = {}
        for p in results['predictions']:
            bg = p['background']
            if bg not in bg_errors:
                bg_errors[bg] = []
            bg_errors[bg].append(p['volume_error'])
        
        backgrounds = list(bg_errors.keys())
        mean_errors = [np.mean(bg_errors[bg]) for bg in backgrounds]
        
        axes[idx].bar(backgrounds, mean_errors, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Background Type', fontsize=12)
        axes[idx].set_ylabel('Mean Absolute Error (mL)', fontsize=12)
        axes[idx].set_title(f'{model_name} - Performance by Background', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(mean_errors):
            axes[idx].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_background.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: performance_by_background.png")
    plt.close()


def create_comparison_table(florence_results: Dict, qwen_results: Dict, output_dir: Path):
    """
    Create a detailed comparison table
    
    Args:
        florence_results: Results from Florence-2
        qwen_results: Results from Qwen2-VL
        output_dir: Directory to save the table
    """
    comparison_data = {
        'Metric': [
            'MAE (mL)',
            'RMSE (mL)',
            'R² Score',
            'Beaker Accuracy (%)',
            'Mean Error (mL)',
            'Std Error (mL)',
            'Max Error (mL)',
            'Min Error (mL)'
        ],
        'Florence-2': [
            f"{florence_results['liquid_volume_metrics']['mae']:.2f}",
            f"{florence_results['liquid_volume_metrics']['rmse']:.2f}",
            f"{florence_results['liquid_volume_metrics']['r2']:.4f}",
            f"{florence_results['beaker_capacity_metrics']['accuracy']*100:.2f}",
            f"{np.mean([p['volume_error'] for p in florence_results['predictions']]):.2f}",
            f"{np.std([p['volume_error'] for p in florence_results['predictions']]):.2f}",
            f"{max([p['volume_error'] for p in florence_results['predictions']]):.2f}",
            f"{min([p['volume_error'] for p in florence_results['predictions']]):.2f}"
        ],
        'Qwen2-VL': [
            f"{qwen_results['liquid_volume_metrics']['mae']:.2f}",
            f"{qwen_results['liquid_volume_metrics']['rmse']:.2f}",
            f"{qwen_results['liquid_volume_metrics']['r2']:.4f}",
            f"{qwen_results['beaker_capacity_metrics']['accuracy']*100:.2f}",
            f"{np.mean([p['volume_error'] for p in qwen_results['predictions']]):.2f}",
            f"{np.std([p['volume_error'] for p in qwen_results['predictions']]):.2f}",
            f"{max([p['volume_error'] for p in qwen_results['predictions']]):.2f}",
            f"{min([p['volume_error'] for p in qwen_results['predictions']]):.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: comparison_table.csv")
    
    # Save as formatted text
    txt_path = output_dir / 'comparison_table.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON: Florence-2 vs Qwen2-VL\n")
        f.write("=" * 70 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 70 + "\n")
    print(f"✓ Saved: comparison_table.txt")
    
    return df


def main():
    """Main evaluation function"""
    print("=" * 70)
    print("BEAKER VOLUME DETECTION - MODEL EVALUATION")
    print("=" * 70)
    
    # Check if models exist
    florence_model_path = FLORENCE_OUTPUT_DIR / "final_model"
    qwen_model_path = QWEN_OUTPUT_DIR / "final_model"
    
    if not florence_model_path.exists():
        print(f"\n❌ Florence-2 model not found at {florence_model_path}")
        print("Please run train_florence.py first!")
        return
    
    if not qwen_model_path.exists():
        print(f"\n❌ Qwen2-VL model not found at {qwen_model_path}")
        print("Please run train_qwen.py first!")
        return
    
    # Load test dataset
    print("\nLoading test dataset...")
    dataset_dict = load_and_split_dataset()
    test_dataset = dataset_dict['test']
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate Florence-2
    print("\n" + "=" * 70)
    print("EVALUATING FLORENCE-2")
    print("=" * 70)
    florence_evaluator = ModelEvaluator(
        str(florence_model_path),
        model_type="florence"
    )
    florence_results = florence_evaluator.evaluate(test_dataset)
    
    # Save Florence-2 results
    with open(RESULTS_DIR / 'florence_results.json', 'w') as f:
        json.dump(florence_results, f, indent=2)
    print(f"\n✓ Florence-2 results saved to {RESULTS_DIR / 'florence_results.json'}")
    
    # Evaluate Qwen2-VL
    print("\n" + "=" * 70)
    print("EVALUATING QWEN2-VL")
    print("=" * 70)
    qwen_evaluator = ModelEvaluator(
        str(qwen_model_path),
        model_type="qwen"
    )
    qwen_results = qwen_evaluator.evaluate(test_dataset)
    
    # Save Qwen2-VL results
    with open(RESULTS_DIR / 'qwen_results.json', 'w') as f:
        json.dump(qwen_results, f, indent=2)
    print(f"\n✓ Qwen2-VL results saved to {RESULTS_DIR / 'qwen_results.json'}")
    
    # Create comparison visualizations
    print("\n" + "=" * 70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 70)
    plot_comparison(florence_results, qwen_results, RESULTS_DIR)
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("CREATING COMPARISON TABLE")
    print("=" * 70)
    comparison_df = create_comparison_table(florence_results, qwen_results, RESULTS_DIR)
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n{comparison_df.to_string(index=False)}\n")
    
    # Determine which model performs better
    florence_mae = florence_results['liquid_volume_metrics']['mae']
    qwen_mae = qwen_results['liquid_volume_metrics']['mae']
    
    print("=" * 70)
    if florence_mae < qwen_mae:
        print("🏆 Florence-2 performs better with lower MAE!")
        improvement = ((qwen_mae - florence_mae) / qwen_mae) * 100
        print(f"   {improvement:.1f}% improvement over Qwen2-VL")
    else:
        print("🏆 Qwen2-VL performs better with lower MAE!")
        improvement = ((florence_mae - qwen_mae) / florence_mae) * 100
        print(f"   {improvement:.1f}% improvement over Florence-2")
    print("=" * 70)
    
    print(f"\n✓ All results saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  • florence_results.json")
    print(f"  • qwen_results.json")
    print(f"  • metrics_comparison.png")
    print(f"  • prediction_scatter.png")
    print(f"  • error_distribution.png")
    print(f"  • performance_by_viewpoint.png")
    print(f"  • performance_by_background.png")
    print(f"  • comparison_table.csv")
    print(f"  • comparison_table.txt")
    
    print("\nNext step:")
    print("  Run demo.py to launch the Gradio interface!")


if __name__ == "__main__":
    main()