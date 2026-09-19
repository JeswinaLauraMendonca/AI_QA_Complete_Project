import json
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "squad_train.jsonl"
MODEL_OUTPUT = PROJECT_ROOT / "data" / "minilm_finetuned"

SEED = 42
MAX_EXAMPLES = 10000

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TOKENIZER FOR BM25 HARD NEGATIVES
# ============================================================

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("ALL-MINILM-L6-V2 FINE-TUNING")
print("=" * 70)

print("\nLoading SQuAD...")

rows = load_jsonl(DATA_FILE)

print(f"Total SQuAD training examples: {len(rows):,}")


# ============================================================
# SAMPLE TRAINING EXAMPLES
# ============================================================

if MAX_EXAMPLES and len(rows) > MAX_EXAMPLES:

    rows = random.sample(rows, MAX_EXAMPLES)

print(f"Examples selected for training: {len(rows):,}")


# ============================================================
# CREATE UNIQUE CONTEXT CORPUS
# ============================================================

contexts = []
context_to_id = {}

for row in rows:

    context = row["context"]
    key = normalize(context)

    if key not in context_to_id:

        context_to_id[key] = len(contexts)
        contexts.append(context)


print(f"Unique contexts: {len(contexts):,}")


# ============================================================
# BUILD BM25 FOR HARD NEGATIVE MINING
# ============================================================

print("\nBuilding BM25 index for hard-negative mining...")

tokenized_contexts = [
    tokenize(context)
    for context in contexts
]

bm25 = BM25Okapi(tokenized_contexts)


# ============================================================
# CREATE QUESTION-POSITIVE PAIRS
# ============================================================

print("\nCreating training pairs...")

examples = []

for row in rows:

    question = row["question"]
    positive_context = row["context"]

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1]

    positive_key = normalize(positive_context)

    # Find difficult BM25 negatives
    negatives = []

    for index in ranked_indices:

        candidate = contexts[int(index)]

        if normalize(candidate) != positive_key:

            negatives.append(candidate)

        if len(negatives) >= 3:

            break

    # --------------------------------------------------------
    # Main positive pair
    # --------------------------------------------------------

    examples.append(
        InputExample(
            texts=[
                question,
                positive_context
            ]
        )
    )


print(f"Training pairs created: {len(examples):,}")


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading all-MiniLM-L6-v2...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 32

train_dataloader = DataLoader(
    examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

train_loss = losses.MultipleNegativesRankingLoss(
    model
)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nTraining device: {device}")

model.to(device)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 3

print("\nStarting fine-tuning...")

print(f"Epochs : {EPOCHS}")
print(f"Batch  : {BATCH_SIZE}")


warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[
        (train_dataloader, train_loss)
    ],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=str(MODEL_OUTPUT),
    show_progress_bar=True
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 70)
print("MINILM TRAINING COMPLETE")
print("=" * 70)

print(f"Fine-tuned model saved to:")
print(MODEL_OUTPUT)

print("=" * 70)