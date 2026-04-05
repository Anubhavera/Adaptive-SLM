"""
MMLU Evaluation for AdaptiveSLM

Evaluates the model on Massive Multitask Language Understanding benchmark
to compare against baselines.
"""

import torch
import json
import argparse
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# MMLU Evaluation
# ============================================================================

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

def format_mmlu_prompt(question: str, choices: List[str], few_shot_examples: str = "") -> str:
    """Format MMLU question as prompt"""
    prompt = few_shot_examples
    prompt += f"Question: {question}\n"
    prompt += "Choices:\n"
    for i, choice in enumerate(choices):
        prompt += f"  ({chr(65 + i)}) {choice}\n"
    prompt += "Answer: ("
    return prompt

def get_few_shot_examples(subject: str, dataset, n_shots: int = 5) -> str:
    """Get few-shot examples from dev set"""
    try:
        dev_data = load_dataset("cais/mmlu", subject, split="dev")
        examples = ""
        
        for i, item in enumerate(dev_data):
            if i >= n_shots:
                break
            
            question = item["question"]
            choices = item["choices"]
            answer = chr(65 + item["answer"])
            
            examples += f"Question: {question}\n"
            examples += "Choices:\n"
            for j, choice in enumerate(choices):
                examples += f"  ({chr(65 + j)}) {choice}\n"
            examples += f"Answer: ({answer})\n\n"
        
        return examples
    except:
        return ""

class MMLUEvaluator:
    """Evaluate model on MMLU benchmark"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_subject(self, subject: str, n_shots: int = 5) -> Dict:
        """Evaluate on a single MMLU subject"""
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"Could not load {subject}: {e}")
            return {"subject": subject, "accuracy": 0.0, "total": 0}
        
        few_shot = get_few_shot_examples(subject, dataset, n_shots)
        
        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc=f"Evaluating {subject}", leave=False):
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            
            prompt = format_mmlu_prompt(question, choices, few_shot)
            
            # Get model prediction
            pred_idx = self._get_prediction(prompt)
            
            if pred_idx == answer_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "subject": subject,
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _get_prediction(self, prompt: str) -> int:
        """Get model's choice prediction (A/B/C/D -> 0/1/2/3)"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get logits for next token
            next_token_logits = outputs.logits[0, -1]
            
            # Get logits for A, B, C, D tokens
            choice_tokens = [
                self.tokenizer.encode("A", add_special_tokens=False)[0],
                self.tokenizer.encode("B", add_special_tokens=False)[0],
                self.tokenizer.encode("C", add_special_tokens=False)[0],
                self.tokenizer.encode("D", add_special_tokens=False)[0],
            ]
            
            choice_logits = [next_token_logits[t].item() for t in choice_tokens]
            
            # Return index of highest logit
            return choice_logits.index(max(choice_logits))
    
    def evaluate_all(self, subjects: List[str] = None, n_shots: int = 5) -> Dict:
        """Evaluate on all MMLU subjects"""
        if subjects is None:
            subjects = MMLU_SUBJECTS
        
        results = []
        total_correct = 0
        total_questions = 0
        
        for subject in tqdm(subjects, desc="MMLU Evaluation"):
            result = self.evaluate_subject(subject, n_shots)
            results.append(result)
            total_correct += result.get("correct", 0)
            total_questions += result.get("total", 0)
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "subject_results": results
        }

# ============================================================================
# Baseline Comparison
# ============================================================================

BASELINE_SCORES = {
    "Qwen2.5-0.5B": 43.7,
    "MobileLLM-125M": 25.6,
    "SmolLM2-360M": 27.3,
    "TinyLlama-1.1B": 26.0,
}

def compare_with_baselines(our_score: float) -> None:
    """Print comparison with baseline models"""
    print("\n" + "=" * 50)
    print("MMLU BENCHMARK COMPARISON")
    print("=" * 50)
    print(f"{'Model':<20} {'MMLU Score':<15} {'Status':<15}")
    print("-" * 50)
    
    # Add our result
    all_scores = {**BASELINE_SCORES, "AdaptiveSLM (ours)": our_score}
    
    # Sort by score
    sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    for model, score in sorted_models:
        status = ""
        if model == "AdaptiveSLM (ours)":
            if score > max(BASELINE_SCORES.values()):
                status = "✓ BEST!"
            elif score > 40:
                status = "✓ Good"
            else:
                status = "Needs work"
        
        print(f"{model:<20} {score:<15.1f} {status:<15}")
    
    print("=" * 50)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveSLM on MMLU")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--n-shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to evaluate")
    parser.add_argument("--output", default="mmlu_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    # Import our model (would need to adjust based on how you saved it)
    from train import AdaptiveSLMModel, AdaptiveSLMConfig, TrainingConfig
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    config = TrainingConfig()
    model_config = AdaptiveSLMConfig(config)
    model = AdaptiveSLMModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Run evaluation
    evaluator = MMLUEvaluator(model, tokenizer)
    results = evaluator.evaluate_all(args.subjects, args.n_shots)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print comparison
    compare_with_baselines(results["overall_accuracy"] * 100)

if __name__ == "__main__":
    main()
