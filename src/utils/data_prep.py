import json, re, random
from typing import Dict, Iterable

# PII Sample
PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d[\d\-\s]{7,}\d"),
}

def redact(text: str) -> str:
    # PII redact logic

def format_example_icl(example: Dict) -> Dict[str, str]:
    """Return a chat-style prompt for instruction tuning (system+user+assistant)."""

    # target : output including actions in steps

    sys = (
        "You are TaskSmith, an assistant that writes precise, numbered, step-by-step "
        "procedures for users. Always produce 5–12 imperative steps, one action per line. "
        "If the user intent is unsafe or impossible, politely refuse and explain why."
    )
    
    # user : input including intent and context
    user = f"INTENT: {intent}\nCONTEXT: {json.dumps(context, ensure_ascii=False)}\n\nWrite the steps."
    return {
        "system": sys,
        "user": user,
        "assistant": target,
    }
