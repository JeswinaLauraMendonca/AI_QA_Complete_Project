import json
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "squad_train.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "bm25_recall_tuning.csv"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# Number of training questions used for tuning.
# Increase later if needed.
MAX_QUESTIONS = 2000

# BM25 configurations
K1_VALUES = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


# ---------------------------------------------------------
# LOAD JSONL
# ---------------------------------------------------------

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# ---------------------------------------------------------
# NORMALIZE CONTEXT
# ---------------------------------------------------------

def normalize_context(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("BM25 RECALL TUNING")
print("=" * 70)

print("\nLoading SQuAD training data...")

rows = load_jsonl(TRAIN_FILE)

print(f"Total training examples: {len(rows):,}")


# ---------------------------------------------------------
# LIMIT QUESTIONS
# ---------------------------------------------------------

if MAX_QUESTIONS is not None and len(rows) > MAX_QUESTIONS:
    rows = rows[:MAX_QUESTIONS]

print(f"Questions used for tuning: {len(rows):,}")


# ---------------------------------------------------------
# CREATE UNIQUE CORPUS
# ---------------------------------------------------------

print("\nCreating BM25 corpus...")

contexts = []
context_to_id = {}

for row in rows:

    context = row["context"]
    normalized = normalize_context(context)

    if normalized not in context_to_id:

        context_id = len(contexts)

        context_to_id[normalized] = context_id
        contexts.append(context)


print(f"Unique contexts: {len(contexts):,}")


# ---------------------------------------------------------
# TOKENIZE CORPUS
# ---------------------------------------------------------

print("\nTokenizing corpus...")

tokenized_contexts = [
    tokenize(context)
    for context in contexts
]


# ---------------------------------------------------------
# MAP EACH QUESTION TO GOLD CONTEXT ID
# ---------------------------------------------------------

gold_context_ids = []

valid_rows = []

for row in rows:

    normalized = normalize_context(row["context"])

    if normalized in context_to_id:

        gold_context_ids.append(
            context_to_id[normalized]
        )

        valid_rows.append(row)


print(f"Valid evaluation questions: {len(valid_rows):,}")


# ---------------------------------------------------------
# TUNE BM25
# ---------------------------------------------------------

results = []

best_recall = -1
best_k1 = None
best_b = None


total_configs = len(K1_VALUES) * len(B_VALUES)

config_number = 0


for k1 in K1_VALUES:

    for b in B_VALUES:

        config_number += 1

        print("\n" + "-" * 70)
        print(
            f"Configuration {config_number}/{total_configs}"
        )
        print(f"k1 = {k1}")
        print(f"b  = {b}")
        print("-" * 70)

        start_time = time.perf_counter()

        # Build BM25 using current parameters
        bm25 = BM25Okapi(
            tokenized_contexts,
            k1=k1,
            b=b
        )

        correct = 0

        total = len(valid_rows)

        for i, row in enumerate(valid_rows):

            question = row["question"]

            gold_id = gold_context_ids[i]

            query_tokens = tokenize(question)

            scores = bm25.get_scores(query_tokens)

            # Best matching context
            predicted_id = int(scores.argmax())

            if predicted_id == gold_id:
                correct += 1

        elapsed = time.perf_counter() - start_time

        recall = correct / total if total > 0 else 0.0

        print(f"Correctly retrieved: {correct}/{total}")
        print(f"Recall: {recall:.4f} ({recall * 100:.2f}%)")
        print(f"Latency: {elapsed:.4f} seconds")

        results.append({
            "k1": k1,
            "b": b,
            "correct": correct,
            "total": total,
            "recall": recall,
            "latency": elapsed
        })

        if recall > best_recall:

            best_recall = recall
            best_k1 = k1
            best_b = b


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BEST BM25 CONFIGURATION")
print("=" * 70)

print(f"k1     : {best_k1}")
print(f"b      : {best_b}")
print(
    f"Recall : {best_recall:.4f} "
    f"({best_recall * 100:.2f}%)"
)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write(
        "k1,b,correct,total,recall,latency\n"
    )

    for result in results:

        f.write(
            f"{result['k1']},"
            f"{result['b']},"
            f"{result['correct']},"
            f"{result['total']},"
            f"{result['recall']:.6f},"
            f"{result['latency']:.6f}\n"
        )


print("\nAll results saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("BM25 TUNING COMPLETE")
print("=" * 70)