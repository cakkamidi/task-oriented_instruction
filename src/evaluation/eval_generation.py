import argparse, json, re, os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer

STEP_LINE = re.compile(r"^\s*\d+\.\s+(.+)$")

def structure_valid(text: str) -> bool:
    return bool(STEP_LINE.search(text))

def normalize(s: str) -> str:

def step_f1(pred: List[str], gold: List[str]) -> float:
    # F1 logic for pred and gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--test_file", type=str, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    # Load test examples and run prediction logic with prompt

    # Metrics
    validity = sum(structure_valid(t) for t in preds) / max(1,len(preds))
    
    # Step F1 avg for all preds and golds

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rls = []
    for pred_text, gold_text in zip(preds, gold_texts):
        rls.append(scorer.score(gold_text, pred_text)["rougeL"].fmeasure)
    rougeL = sum(rls)/max(1,len(rls))


    report = {
        "structure_valid_rate": validity,
        "avg_step_f1": avg_f1,
        "rougeL_f1": rougeL,
        "bertscore_f1": berts,
        "n": len(preds),
    }