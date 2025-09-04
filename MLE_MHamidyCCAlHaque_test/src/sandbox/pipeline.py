import argparse, json, re
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_steps(mdl, tok, intent: str, context: Dict, max_new_tokens=256) -> str: #adjust max new tokens as desired
    # generate with prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--intent", type=str, default="How do I reset my password in the e-commerce 'xx' mobile app?")
    ap.add_argument("--context", type=str, default='{"platform":"android","app":"xx","locale":"en_ID"}')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    steps_text = generate_steps(mdl, tok, args.intent, json.loads(args.context))
