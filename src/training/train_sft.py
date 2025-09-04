import argparse, json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from utils.data_prep import format_example_icl
import pandas as pd

def make_dataset(path):
    # get from format_example_icl function in data_prep.py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs/model")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    dataset = make_dataset(args.train_file)
    from datasets import Dataset
    ds = Dataset.from_list(dataset)

    # Tokenizer & 4-bit base

    base = AutoModelForCausalLM.from_pretrained(
        #params
    )

    lora = LoraConfig(
        #params
    )
    model = get_peft_model(base, lora)

    #SFT config definition

    trainer = SFTTrainer(
        model=model,
        #params
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)