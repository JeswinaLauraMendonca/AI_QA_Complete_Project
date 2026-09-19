import json
import random
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "squad_train.jsonl"
MODEL_PATH = PROJECT_ROOT / "data" / "minilm_finetuned"

OUTPUT_FILE = PROJECT_ROOT / "data" / "minilm_recall_evaluation.csv"


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Use a separate subset from the 10,000 examples used for
# MiniLM training.
EVAL_SIZE = 2000

random.seed(SEED)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
print("MINILM RECALL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading SQuAD training data...")

rows = load_jsonl(TRAIN_FILE)

print(f"Total training examples: {len(rows):,}")


# ------------------------------------------------------------
# IMPORTANT:
# The MiniLM training script used random.seed(42) and
# random.sample(..., 10000).
#
# We reproduce that same selection so we can exclude those
# training examples from evaluation.
# ------------------------------------------------------------

all_indices = list(range(len(rows)))

random.seed(SEED)

training_indices = set(
    random.sample(
        all_indices,
        10000
    )
)


remaining_indices = [
    i for i in all_indices
    if i not in training_indices
]


# Select a separate evaluation set
random.seed(SEED)

evaluation_indices = random.sample(
    remaining_indices,
    min(EVAL_SIZE, len(remaining_indices))
)


eval_rows = [
    rows[i]
    for i in evaluation_indices
]


print(
    f"MiniLM training examples excluded: "
    f"{len(training_indices):,}"
)

print(
    f"Evaluation questions: "
    f"{len(eval_rows):,}"
)


# ============================================================
# BUILD EVALUATION CORPUS
# ============================================================

print("\nBuilding evaluation corpus...")

contexts = []
context_to_id = {}

for row in eval_rows:

    context = row["context"]

    key = normalize(context)

    if key not in context_to_id:

        context_to_id[key] = len(contexts)

        contexts.append(context)


print(f"Unique evaluation contexts: {len(contexts):,}")


# ============================================================
# CREATE GOLD CONTEXT IDS
# ============================================================

gold_ids = []

questions = []

for row in eval_rows:

    question = row["question"]

    context = row["context"]

    key = normalize(context)

    gold_id = context_to_id[key]

    questions.append(question)

    gold_ids.append(gold_id)


# ============================================================
# LOAD FINE-TUNED MINILM
# ============================================================

print("\nLoading fine-tuned MiniLM...")

model = SentenceTransformer(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nEncoding evaluation contexts...")

start = time.perf_counter()

context_embeddings = model.encode(
    contexts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32
)

context_time = time.perf_counter() - start

print(
    f"Context encoding time: "
    f"{context_time:.2f} seconds"
)


# ============================================================
# ENCODE QUESTIONS
# ============================================================

print("\nEncoding questions...")

start = time.perf_counter()

question_embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32
)

question_time = time.perf_counter() - start

print(
    f"Question encoding time: "
    f"{question_time:.2f} seconds"
)


# ============================================================
# RETRIEVAL
# ============================================================

print("\nEvaluating retrieval...")

correct = 0

total_latency = 0.0

for i, question_embedding in enumerate(question_embeddings):

    start = time.perf_counter()

    # Because embeddings are normalized, dot product
    # is cosine similarity.
    scores = np.dot(
        context_embeddings,
        question_embedding
    )

    predicted_id = int(
        np.argmax(scores)
    )

    latency = time.perf_counter() - start

    total_latency += latency

    if predicted_id == gold_ids[i]:

        correct += 1


# ============================================================
# FINAL RECALL
# ============================================================

total = len(gold_ids)

recall = correct / total if total else 0.0

average_latency = (
    total_latency / total
    if total
    else 0.0
)


# ============================================================
# SAVE RESULT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "model,correct,total,recall,average_latency\n"
    )

    f.write(
        f"MiniLM,"
        f"{correct},"
        f"{total},"
        f"{recall:.6f},"
        f"{average_latency:.6f}\n"
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL MINILM RECALL RESULT")
print("=" * 70)

print(
    f"Correctly retrieved : "
    f"{correct}/{total}"
)

print(
    f"Recall              : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"Average retrieval latency : "
    f"{average_latency:.6f} seconds"
)

print("\nResult saved to:")

print(OUTPUT_FILE)

print("=" * 70)
print("MINILM EVALUATION COMPLETE")
print("=" * 70)