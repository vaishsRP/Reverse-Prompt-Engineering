import pandas as pd
print(pd.read_csv('data/reconstructions_t0.csv').shape)
import pandas as pd
import numpy as np
import re
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ── Load data ──────────────────────────────────────────────────────────────────
t0  = pd.read_csv("data/reconstructions_t0.csv")
t07 = pd.read_csv("data/reconstructions_t0.7.csv")

merged = t0.merge(
    t07[["id", "reconstructed_prompt"]],
    on="id",
    suffixes=("_t0", "_t07")
)

# ── Helpers ────────────────────────────────────────────────────────────────────
STOPWORDS = {
    "that", "this", "with", "have", "from", "they", "been", "were",
    "what", "when", "your", "their", "about", "some", "just", "would",
    "could", "should", "really", "feel", "think", "know", "like",
    "want", "need", "help", "make", "also", "into", "more", "very"
}

def extract_keywords(text):
    tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def keyword_recall(original, reconstruction):
    kws = extract_keywords(original)
    if not kws:
        return 0.0
    recon_tokens = set(re.findall(r"\b[a-z]{4,}\b", reconstruction.lower()))
    hits = sum(1 for kw in kws if kw in recon_tokens)
    return hits / len(kws)

def jaccard(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 | set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def word_count(text):
    return len(str(text).split())

# ── ROUGE ──────────────────────────────────────────────────────────────────────
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def rouge_scores(reference, hypothesis):
    s = rouge.score(reference, hypothesis)
    return s["rouge1"].fmeasure, s["rougeL"].fmeasure

# ── Compute row-level metrics ──────────────────────────────────────────────────
results = []

for _, row in merged.iterrows():
    original   = str(row["prompt"])
    recon_t0   = str(row["reconstructed_prompt_t0"])
    recon_t07  = str(row["reconstructed_prompt_t07"])

    r1_t0,  rL_t0  = rouge_scores(original, recon_t0)
    r1_t07, rL_t07 = rouge_scores(original, recon_t07)

    results.append({
        "id":            row["id"],
        "style":         row["style"],
        "level":         row["level"],
        "topic":         row["topic"],
        "rouge1_t0":     round(r1_t0, 4),
        "rouge1_t07":    round(r1_t07, 4),
        "rougeL_t0":     round(rL_t0, 4),
        "rougeL_t07":    round(rL_t07, 4),
        "kw_recall_t0":  round(keyword_recall(original, recon_t0), 4),
        "kw_recall_t07": round(keyword_recall(original, recon_t07), 4),
        "len_ratio_t0":  round(word_count(recon_t0) / max(word_count(original), 1), 4),
        "len_ratio_t07": round(word_count(recon_t07) / max(word_count(original), 1), 4),
        "t0_t07_overlap": round(jaccard(recon_t0, recon_t07), 4),
        # placeholders for human annotation — fill manually
        "human_score_t0":  None,
        "human_score_t07": None,
        "error_type_t0":   None,
        "error_type_t07":  None,
    })

results_df = pd.DataFrame(results)

# ── BERTScore ──────────────────────────────────────────────────────────────────
# Run separately because it loads a model — slow but only runs once
print("Running BERTScore t0...")
_, _, F1_t0 = bert_score_fn(
    list(results_df["id"].map(dict(zip(merged["id"], merged["reconstructed_prompt_t0"])))),
    list(t0["prompt"]),
    lang="en",
    verbose=False
)

print("Running BERTScore t0.7...")
_, _, F1_t07 = bert_score_fn(
    list(results_df["id"].map(dict(zip(merged["id"], merged["reconstructed_prompt_t07"])))),
    list(t0["prompt"]),
    lang="en",
    verbose=False
)

results_df["bert_f1_t0"]  = np.round(F1_t0.numpy(), 4)
results_df["bert_f1_t07"] = np.round(F1_t07.numpy(), 4)

# ── BERTScore reconstruction vs output ──────────────────────────────────
print("Running BERTScore reconstruction vs output...")
_, _, F1_recon_vs_output = bert_score_fn(
    list(results_df["id"].map(dict(zip(merged["id"], merged["reconstructed_prompt_t0"])))),
    list(t0["output"]),
    lang="en",
    verbose=False
)

results_df["bert_f1_recon_vs_output"] = np.round(F1_recon_vs_output.numpy(), 4)

# ── Save ───────────────────────────────────────────────────────────────────────
import os
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/evaluation_results.csv", index=False)
print("Saved results/evaluation_results.csv")

# ── Summary stats ──────────────────────────────────────────────────────────────
metrics = ["rouge1", "rougeL", "bert_f1", "kw_recall", "len_ratio"]
print("\n── Overall means ──")
for m in metrics:
    t0_mean  = results_df[f"{m}_t0"].mean()
    t07_mean = results_df[f"{m}_t07"].mean()
    print(f"{m:15} t0={t0_mean:.3f}  t0.7={t07_mean:.3f}")

print("\n── By style ──")
for style in results_df["style"].dropna().unique():
    sub = results_df[results_df["style"] == style]
    print(f"\n{style}")
    for m in metrics:
        print(f"  {m:15} t0={sub[f'{m}_t0'].mean():.3f}  t0.7={sub[f'{m}_t07'].mean():.3f}")

print("\n── By level ──")
for level in results_df["level"].dropna().unique():
    sub = results_df[results_df["level"] == level]
    print(f"\n{level}")
    for m in metrics:
        print(f"  {m:15} t0={sub[f'{m}_t0'].mean():.3f}  t0.7={sub[f'{m}_t07'].mean():.3f}")

print("\n── By topic ──")
for topic in results_df["topic"].dropna().unique():
    sub = results_df[results_df["topic"] == topic]
    print(f"\n{topic}")
    for m in ["bert_f1", "kw_recall"]:
        print(f"  {m:15} t0={sub[f'{m}_t0'].mean():.3f}  t0.7={sub[f'{m}_t07'].mean():.3f}")

print("\n── Jaccard t0 vs t0.7 ──")
print(f"  mean={results_df['t0_t07_overlap'].mean():.3f}  min={results_df['t0_t07_overlap'].min():.3f}  max={results_df['t0_t07_overlap'].max():.3f}")
print(f"  rows with overlap < 0.30: {(results_df['t0_t07_overlap'] < 0.30).sum()}")

print("\n── BERTScore reconstruction vs output ──")
print(f"  mean={results_df['bert_f1_recon_vs_output'].mean():.3f}  min={results_df['bert_f1_recon_vs_output'].min():.3f}  max={results_df['bert_f1_recon_vs_output'].max():.3f}")

print("\nDone.")