1) Open‑Source LLM Choice

With the objective of generating structured task-oriented instructions, the model selection should consider strong capabilities of : reasoning & instruction-following.
Here are provided 3 relevant opensource models to consider.

- Llama 3 : RL with human feedback on post-training, massive high-quality training data, and expanded context window of 8192 tokens

Best for reliability and broad, general-purpose instruction generation.
Limitations of high resource cost, risk of catastrophic forgetting, and data dependency.

- Mistral : MoE activating best suited experts, reasoning-specific Magistral fine-tuned for multi-step logic

Best for efficiency, speed, and cost-effectiveness for generating a high volume of instructions.
Limitations of robustness, complex fine-tuning architecture for each expert, and inference latency.

- Deepseek : Group Relative Policy Optimization, Chain-of-Thought, verification loops

Best for highly complex or technically specified instructions where logical accuracy is desired.
Limitations of RL-specific, readability of CoT, and less general-purpose.


As a conclusion, Llama 3 is preferred due to the considerable reliability and the more well-understood ways to tackle its limitations, such as using smaller versions instead which already offer a strong performance-to-cost balance, and implementing fine-tuning approaches which is not too complicated.


2) Dataset Design & Preparation

The data to fine‑tune is in the form of pairs of intent & structured procedure. Each example is a JSON object with:
- `input.intent`: natural‑language intent (free text).
- `input.context`: optional app, platform, locale, user role, permissions, current screen, version.
- `output.schema`: always `"ordered_steps"` (extensible to `"checklist"`, `"flowchart"`).
- `output.steps`: ordered list of atomic actions (imperative mood, 1 action/step).
- `output.constraints`: preconditions, security/privacy notes, edge cases.
- `output.validation`: simple checks the user can do to confirm success.
- `meta`: domain tags (e.g., ecommerce/passwords), difficulty, language, annotator id, source.

Examples are provided in `/src/data/sample_train.jsonl`.

Hypothetically, data collection and annotation can be performed as below.
1. Scrap from public product docs/FAQs, UX flows, and internal runbooks.  
2. Crowd/expert annotation by giving raters a UI to read intent + app context before writing steps (≤12), constraints, and validation checks. Enforce style guide (tep granularity, safety).  
3. Synthetic bootstrapping by prompting a strong model to draft procedures while humans performing review & edit (“human‑in‑the‑loop”).  

Preprocessing of this project is performed as below.
- Normalization: lowercase app artifacts where appropriate, strip boilerplate, minimalize near‑duplicates.
- Schema enforcement: validate against `schema.json`.
- PII redaction: emails/phones/order IDs masked (`<EMAIL>`, `<PHONE>`, `<ORDER_ID>`).  
- Formatting to training text: pack into an instruction‑tuning template with a "system" section that states the schema, then "user" (intent+context), "assistant" (JSON or numbered list). See `utils/data_prep.py`.

Accomodating self‑correction & edge cases
- Critique‑and‑revise: generate draft → run a self‑check prompt that verifies style rules (numbered, imperative, <=N steps, contains validation). If failed, regenerate with constraints.  
- Constraint‑guided decoding: use a regex/JSON‑schema validator; if invalid, block‑sample until valid.  
- Imbalance: stratified sampling + loss reweighting by domain.
- Sensitive info: strict PII redaction, domain allow‑lists, and reject/label intents that seek unsafe instructions.  
- Generalization: include counter‑examples (e.g., “no ‘Forgot Password’ present”), multilingual variants, and recovery flows for cases.


3) Fine‑Tuning Strategy

For fine-tuning, PEFT-based instruction tuning such as LoRA/QLoRA is preferred due to its efficiency by training only small low-rank matrices instead, modularity by swapping multiple adapters for different domains, and low risk of catastrophic forgetting. Full fine-tuning is not preferred for the extremely high cost and risk of catastrophic forgetting. Also, other approaches which are supervised would require human annotators or reliable preference dataset.

Key hyperparameters to tune include the followings.
- r (rank) (LoRA's) : r value to be tuned to allow task-specific adaptations while considering memory usage.
- alpha (scaling factor) (LoRA's) : balancing influence of fine-tuned parameters vs base model knowledge.
- target_modules (LoRA's) : LoRA adapters are injected to modules in the transformer, common practice in attention layers, more coverage may have risks of computational cost.
- common training hyperparameters : learning rate, batch size, epochs/training steps (early stopping), warmup ratio (early stopping), max sequence length.
- regularization/stability : dropout, gradient clipping.
- decoding (eval): temperature, top_p, length penalty.

Risks & mitigations include the followings.
- Overfitting: early stop; mix in a small portion of general instruct data.  
- Catastrophic forgetting: PEFT isolates adapters; optionally train with 5–10% base general data.  
- Computational resources: QLoRA + 8B/7B base; gradient checkpointing.  
- License/usage: confirm Llama Community License terms or might consider Apache‑2.0 (Mistral) for simpler ops.

4) Evaluation & Benchmarking

Automatic metrics (quantitative) :
- Structure validity rate: percentage outputs that pass JSON/regex schema.  
- Step F1: match gold steps by normalized verbs+objects (synonym aware).  
- ROUGE‑L / BLEU / BERTScore: lexical + semantic similarity to references.  
- Constraint satisfaction: percentage required constraints appearance.  
- Hallucination rate: percentage steps mentioning out‑of‑context entities (via allow‑lists).

Human evaluation (qualitative) : 5‑point Likert on clarity, completeness, correctness, safety; pairwise win‑rate vs. baseline.  

**Benchmarking setup**
- Hold‑out test split with diverse domains/locales.  
- Baselines: base LLM (zero‑shot), rule‑based templates, and human‑written gold.  
- Report both overall and per‑domain scores; include ablations (no self‑check, no constraints).

---


IMPLEMENTATION

`/src`:
- `training/train_sft.py` — QLoRA SFT with TRL/PEFT.  
- `utils/data_prep.py` — schema validation, redaction, prompt formatting.  
- `evaluation/eval_generation.py` — structure checks + ROUGE/BERTScore hooks.  
- `inference/pipeline.py` — inference wrapper returning numbered steps + JSON.  
- `data/schema.json` and `data/sample_train.jsonl` — schema and tiny sample.


## Quick Start (local)
```bash
# Create env (Python >=3.10)
pip install -U transformers datasets accelerate peft trl bitsandbytes rouge-score bert-score fastapi pydantic uvicorn

# Train (QLoRA SFT)
python src/training/train_sft.py --base_model meta-llama/Llama-3.1-8B-Instruct --train_file src/data/sample_train.jsonl --output_dir ./outputs/llama3.1-8b-instruct-task

# Evaluate
python src/evaluation/eval_generation.py --model ./outputs/llama3.1-8b-instruct-task --test_file src/data/sample_train.jsonl

# Serve (sandbox)
python src/inference/pipeline.py --model ./outputs/llama3.1-8b-instruct-task
```
