# Reverse Prompt Reconstruction from LLM-Generated Personal Advice

**NLP Assignment 3 — VU Amsterdam 2026**  
Priyanshi Dhillon · Shithik Shaji · Vaishanavi Mehta

---

## Overview

This project implements an end-to-end **reverse prompt engineering pipeline**. Given a piece of LLM-generated personal advice, can we recover the original human situation that prompted it?

The pipeline has three stages:
1. **Generate** — Feed 50 hand-crafted personal advice prompts (tagged with Big Five personality style) to `llama-3.3-70b-versatile` to produce a gold dataset of advice responses.
2. **Reconstruct** — Ask the same model to infer the original human prompt from each advice response alone, at two temperatures (t=0 and t=0.7).
3. **Evaluate** — Score each reconstruction against the original prompt using ROUGE, BERTScore, keyword recall, and length ratio. A diagnostic BERTScore (reconstruction vs. advice output) is also computed to distinguish genuine inversion from mere paraphrase.

---

## Project Structure

```
.
├── data/
│   ├── prompts.csv                # 50 hand-crafted input prompts (ID, Topic, Style, Level, Prompt)
│   ├── gold_dataset.csv           # LLM-generated advice responses for each prompt
│   ├── reconstructions_t0.csv     # Reconstructed prompts at temperature=0 (deterministic)
│   ├── reconstructions_t0.7.csv   # Reconstructed prompts at temperature=0.7 (stochastic)
│   ├── prompt.py                  # Answer_prompt and Reverse_prompt templates
│   └── __init__.py
├── src/
│   ├── generate.py                # Stage 1: generate gold dataset from prompts
│   ├── reconstruct.py             # Stage 2: reconstruct original prompts from advice
│   └── evaluate.py                # Stage 3: compute all evaluation metrics
├── results/
│   └── evaluation_results.csv     # Pre-computed evaluation scores (all 50 rows, all metrics)
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone / unzip the project

```bash
git clone https://github.com/PriyanshiDhillon/Reverse-Prompt-Engineering.git
```
OR
```bash
unzip NAMES_A3_NLP2026.zip
cd Reverse-Prompt-Engineering
```

### 2. Install dependencies

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

This project uses the [Groq API](https://console.groq.com/) to access `llama-3.3-70b-versatile` via an OpenAI-compatible endpoint. A free Groq account gives sufficient quota to reproduce all results.

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
LLM_API_KEY = "your_groq_api_key_here"
LLM_API_BASE_URL = "https://api.groq.com/openai/v1"
```

---

## Reproducing the Results

All three scripts must be run from the **project root directory** (not from inside `src/`), so that the `data/` imports resolve correctly.

### Stage 1 — Generate advice responses

Reads `data/prompts.csv` and writes `data/gold_dataset.csv`.  
**Skip this step** if you want to use the pre-generated data already included in the zip.

```bash
python src/generate.py
```

- Model: `llama-3.3-70b-versatile`
- Temperature: `0.7`
- Max tokens: `300`
- A 0.6-second delay between calls prevents rate-limit errors on the Groq free tier.

---

### Stage 2 — Reconstruct original prompts

Reads `data/gold_dataset.csv` and writes reconstruction CSVs.  
**Skip this step** if you want to use the pre-generated reconstructions already in the zip.

Run at **t=0** (deterministic):
```bash
python -c "
from src.reconstruct import run_reconstruction
run_reconstruction('data/gold_dataset.csv', 'data/reconstructions_t0.csv', temperature=0.0)
"
```

Run at **t=0.7** (stochastic):
```bash
python src/reconstruct.py
```

- A 2.5-second delay between calls is included to stay within Groq free-tier rate limits.
- Any API errors are caught and written as `"ERROR"` in the output CSV so the run does not abort.

---

### Stage 3 — Evaluate

Reads `data/reconstructions_t0.csv` and `data/reconstructions_t0.7.csv`, computes all metrics, and writes `results/evaluation_results.csv`.

```bash
python src/evaluate.py
```

This will also print a full breakdown to stdout: overall means, by style, by level, by topic, Jaccard overlap between temperature runs, and the diagnostic BERTScore (reconstruction vs. advice output).

> **Pre-computed results** are already provided in `results/evaluation_results.csv`, so you can inspect the scores without re-running the API calls.

---

## Data Format

### `data/prompts.csv`
| Column | Description |
|--------|-------------|
| `ID` | Unique integer identifier (1–50) |
| `Topic` | One of: Relationship, Workplace Sabotage, Friendship & Social Circles, Digital & Social Media, Family Dynamics |
| `Style` | Big Five personality style: `Openness` or `Conscientiousness` |
| `Level` | Strength of the style: `High` or `Low` |
| `Prompt` | First-person human advice prompt (1–3 sentences, ends with a question) |

### `data/gold_dataset.csv`
Same as above plus:
| Column | Description |
|--------|-------------|
| `output` | LLM-generated advice response (4–6 sentences, styled per Style/Level) |

### `data/reconstructions_t0.csv` / `reconstructions_t0.7.csv`
Same as gold dataset plus:
| Column | Description |
|--------|-------------|
| `reconstructed_prompt` | Model's best guess at the original human prompt |
| `temperature` | Decoding temperature used (0 or 0.7) |

### `results/evaluation_results.csv`
One row per prompt (50 rows). Columns:

| Column | Description |
|--------|-------------|
| `rouge1_t0` / `rouge1_t07` | ROUGE-1 F1 (unigram overlap) vs. original prompt |
| `rougeL_t0` / `rougeL_t07` | ROUGE-L F1 (longest common subsequence) vs. original prompt |
| `bert_f1_t0` / `bert_f1_t07` | BERTScore F1 (RoBERTa-large) vs. original prompt |
| `kw_recall_t0` / `kw_recall_t07` | Fraction of content keywords from original prompt found in reconstruction |
| `len_ratio_t0` / `len_ratio_t07` | Reconstruction word count / original prompt word count |
| `t0_t07_overlap` | Jaccard word overlap between t=0 and t=0.7 reconstructions |
| `bert_f1_recon_vs_output` | **Diagnostic:** BERTScore of reconstruction vs. advice output (not vs. original prompt) — used to detect paraphrase vs. genuine inversion |

---

## Key Design Decisions

**Why `llama-3.3-70b-versatile`?**  
At 70B parameters it produces stylistically rich advice outputs, making the reconstruction task genuinely non-trivial. Smaller models (7B/13B) produce outputs with little variation across personality conditions.

**Why Groq?**  
Free tier, low latency, OpenAI-compatible endpoint — easy to reproduce without cost.

**Why two temperatures?**  
t=0 gives a reproducible deterministic baseline. t=0.7 reintroduces stochasticity to test whether higher temperature helps the model explore alternative valid reconstructions. The near-identical BERTScore across both (0.890 vs. 0.889) confirms the performance ceiling is set by the information bottleneck in the advice genre, not by decoding strategy.

**Why the diagnostic BERTScore?**  
Standard metrics alone cannot distinguish a model that genuinely recovers the original prompt from one that merely paraphrases the advice text it was given as input. The 0.023 gap between reconstruction-vs-prompt (0.890) and reconstruction-vs-output (0.867) is the central finding: the system compresses rather than inverts.

**Why keyword recall alongside BERTScore?**  
BERTScore captures semantic similarity but is insensitive to whether specific concrete details survive (e.g. "debt collector" vs. "financial stress"). Keyword recall is a direct measure of situational completeness and reveals the dissociation that BERTScore masks.

---

## Summary of Results

| Metric | t=0 | t=0.7 |
|--------|-----|-------|
| ROUGE-1 | 0.340 | 0.336 |
| ROUGE-L | 0.217 | 0.214 |
| BERTScore F1 (vs. prompt) | 0.890 | 0.889 |
| Keyword Recall | 0.265 | 0.260 |
| Length Ratio | 1.663 | 1.652 |
| BERTScore (vs. output) | 0.867 | — |
| **Diagnostic gap** | **0.023** | — |

The small diagnostic gap (0.023) is the key result: the model is largely paraphrasing the advice it received rather than recovering the original human situation.

---

## Dependencies

```
bert-score==0.3.13
rouge-score==0.1.2
openai==2.36.0
python-dotenv==1.2.2
tqdm==4.67.3
pandas==3.0.2
numpy==2.4.4
```
