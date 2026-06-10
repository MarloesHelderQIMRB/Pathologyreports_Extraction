"""
Fine-tuning LLaMA-3.1-8B-Instruct for Pathology Report Classification

Fine-tunes Meta-Llama-3.1-8B-Instruct to extract skin lesion information from pathology reports: lesion count, diagnosis, anatomical site, and facial location.

Usage:
    python train_qskin_llama.py --local_model_path /path/to/llama-3.1-8b-Instruct \
                                --model_save_path /path/to/output \
                                --training_data_path /path/to/data

Data format training and validation files: CSV with columns [system_prompt, user_prompt, assistant_output]. The system and user prompts can be found in the paper.
"""

# Load packages
import argparse
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def setup_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3.1-8B for pathology reports")
    parser.add_argument("--local_model_path", required=True, help="Path to locally downloaded LLaMA model")
    parser.add_argument("--model_save_path", required=True, help="Output directory for trained model")
    parser.add_argument("--training_data_path", required=True, help="Directory with train.csv and validation.csv")
    parser.add_argument("--run_name", required=True, help="Training run name")
    return parser.parse_args()


def load_model(model_path):
    """Load and quantize LLaMA model with LoRA configuration."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=None
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    return tokenizer, model

def load_data(data_path):
    """Load and format training and validation data."""
    train_df = pd.read_csv(f"{data_path}/train.csv")
    val_df = pd.read_csv(f"{data_path}/validation.csv")
    
    def create_conversations_list(row):
        return [
            {"from": "system", "value": row['system_prompt']},
            {"from": "human", "value": row['user_prompt']},
            {"from": "gpt", "value": row['assistant_output']}
        ]

    train_df['conversations'] = train_df.apply(create_conversations_list, axis=1)
    train_df2 = Dataset.from_pandas(train_df)

    val_df['conversations'] = val_df.apply(create_conversations_list, axis=1)
    val_df2 = Dataset.from_pandas(val_df)

    def format_chat(example):
        text = (
            f"<SYSTEM> {example['system_prompt']}\n"
            f"<HUMAN> {example['user_prompt']}\n"
            f"<ASSISTANT> {example['assistant_output']}"
        )
        return {"text": text}

    train_dataset = train_df2.map(format_chat)
    val_dataset = val_df2.map(format_chat)
    
    return train_dataset, val_dataset

def main():
    args = setup_args()
    os.makedirs(args.model_save_path, exist_ok=True)
    
    tokenizer, model = load_model(args.local_model_path)
    train_dataset, val_dataset = load_data(args.training_data_path)
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Training configuration
    training_config = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=4,
        packing=False,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        logging_steps=15,
        eval_steps=250,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=250,
        optim="adamw_bnb_8bit",
        save_total_limit=5,
        weight_decay=0.05,
        warmup_ratio=0.03,
        output_dir=args.model_save_path,
        group_by_length=False,
        run_name=args.run_name,
        resume_from_checkpoint=True
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_config
    )
    
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(args.model_save_path)
    print(f"Model saved to: {args.model_save_path}")


if __name__ == "__main__":
    main()