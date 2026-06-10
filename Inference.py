""""
Finetuned LLaMA-3.1-8B-Instruct Inference Script

Generates predictions from the fine-tuned llama model on test data.
Runs multiple repetitions and saves timestamped CSV files with predictions.

Usage:
    python validate_qskin_llama.py --model_save_path /path/to/trained/model \
                                   --adapter_path /path/to/checkpoint-X \
                                   --test_set_loc /path/to/test.csv \
                                   --output_dir /path/to/predictions

Output: CSV files with columns [uid, text, preds] where preds contains JSON predictions
"""

import argparse
import os
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def setup_args():
    parser = argparse.ArgumentParser(description="Generate predictions from fine-tuned llama model")
    parser.add_argument("--model_save_path", required=True, help="Path to base model directory")
    parser.add_argument("--adapter_path", required=True, help="Path to fine-tuned LoRA adapter (checkpoint)")
    parser.add_argument("--test_set_loc", required=True, help="Path to test CSV file") 
    parser.add_argument("--output_dir", required=True, help="Directory to save prediction CSV files")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of prediction runs (default: 5)")
    return parser.parse_args()


def load_model(base_model_path, adapter_path):
    """Load base model with fine-tuned LoRA adapter."""
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=None
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def get_system_prompt():
    """Return the system prompt with allowed diagnoses and sites."""
    return """
    You are an expert pathology AI assistant. Your task is to extract all skin lesion information from the clinical text provided. Only assign diagnoses and site from the set of options below.

    Allowed diagnoses: 
    'Squamous cell carcinoma (SCC)', 'scc re-excision - clear', 'Basal cell carcinoma (BCC)', 'bcc re-excision - clear', 'precursor lesions', 'benign naevus', 'Bowens disease', 'Bowens disease giving rise to SCC', 'Dysplastic naevus', 'Intraepidermal carcinoma (IEC)', 'iec re-excision - clear', 'Keratoacanthoma (KA)', 'Lentigo maligna', 'Lentigo/solar lentigo', 'Melanoma', 'Non-malignant lesion', 'No skin lesions', 'Other', 'Seborrhoeic keratosis', 'Solar keratosis', 'Squamo-proliferative lesions'.

    Allowed sites: 
    'Abdomen', 'Back/ NOS', 'Back Of Hand', 'Breast', 'Buttock', 'Ears', 'Face', 'Finger nail', 'Forearm, Elbow, Wrist', 'Hip', 'Illegible', 'Lower back', 'Lower Leg, Ankle, Knee', 'Neck', 'Non-Skin', 'No record', 'Palmar Skin, Fingers', 'Perineum', 'Plantar Skin, Toes', 'Scalp', 'Shoulders', 'Site', 'Skin/NOS', 'Thigh', 'Top of feet', 'Trunk/NOS', 'UpperArm', 'Upper back', 'Upper chest/ Sternoclavicular'.

    Allowed sites on the face:
    'Cheeks', 'Chin/ Jaw', 'Face/NOS', 'Forehead', 'Lips', 'Nose', 'Siteface', 'Skin of orbit/ eyelid', 'Temple'.

    Instructions:
    1. Identify the diagnosis and site of ALL lesions mentioned in the text.
    2. Assign a unique numeric ID to each lesion.
    3. For each lesion, extract the following fields:
      - "Diagnosis": primary diagnosis (must be from the allowed set)
     - "Site": primary site (must be from allowed sites)
      - "SiteFace": if site is 'Face', give specific location on face, otherwise leave as an empty string (must be from the allowed sites on the face)
    4. Output strictly as a valid JSON object with the following format:

    {"1":{"Diagnosis":"..","Site":"..","SiteFace":".."},"2":{"Diagnosis":"..","Site":"..","SiteFace":".."},..}
    """.strip()


def generate_predictions(model, tokenizer, reports, system_prompt):
    """Generate predictions for a list of pathology reports."""
    predictions = []
    
    for report in tqdm(reports, desc="Generating predictions"):
        try:
            user_content = ("What is Diagnosis, Site, and SiteFace for each lesion from this text? Output it as a JSON object, just generate the JSON object without explanations. \n" + report)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Apply chat template and generate
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            
            output = model.generate(
                input_ids=inputs, 
                max_new_tokens=500, 
                use_cache=True
            )
            
            # Decode response
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            response = decoded[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            predictions.append(response.strip())
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error processing report: {e}")
            predictions.append("ERROR: Processing failed")
    
    return predictions

def run_validation(model, tokenizer, test_df, output_dir, repetitions):
    """Run validation for specified number of repetitions."""
    os.makedirs(output_dir, exist_ok=True)
    system_prompt = get_system_prompt()
    
    for i in range(repetitions):
        print(f"Starting Run {i+1}/{repetitions}")
        start_time = time.time()
        
        # Generate predictions
        predictions = generate_predictions(
            model, tokenizer, test_df['text'], system_prompt
        )
        
        # Calculate timing statistics
        end_time = time.time()
        elapsed = end_time - start_time
        n_reports = len(test_df)
        reports_per_min = (n_reports / elapsed) * 60
        
        print(f"Run {i+1} completed:")
        print(f"  Reports processed: {n_reports}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print(f"  Reports per minute: {reports_per_min:.1f}")
        print(f"  Avg time per report: {elapsed/n_reports:.2f} seconds")
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{output_dir}/predictions-{timestamp}.csv"
        
        results_df = pd.DataFrame({
            "uid": test_df['uid'],
            "text": test_df['text'],
            "preds": predictions
        })
        results_df.to_csv(filename, index=False)


def main():
    args = setup_args()
    test_df = pd.read_csv(args.test_set_loc)
    print(f"Test samples: {len(test_df)}")
    
    model, tokenizer = load_model(args.model_save_path, args.adapter_path)
    
    run_validation(model, tokenizer, test_df, args.output_dir, args.repetitions)
    
    print(f"Prediction files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()