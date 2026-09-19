import json
import pickle
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "squad_train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "squad_validation.jsonl"
INDEX_FILE = PROJECT_ROOT / "data" / "index.pkl"


def normalize(text):
    """Normalize text so equivalent contexts can be compared."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


print("=" * 70)
print("SQUAD TRAIN / VALIDATION CONTEXT OVERLAP CHECK")
print("=" * 70)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading SQuAD files...")

train_rows = load_jsonl(TRAIN_FILE)
val_rows = load_jsonl(VAL_FILE)

print(f"Training examples   : {len(train_rows):,}")
print(f"Validation examples : {len(val_rows):,}")


# ---------------------------------------------------------
# Extract contexts
# ---------------------------------------------------------

train_contexts = set(
    normalize(row["context"])
    for row in train_rows
)

val_contexts = set(
    normalize(row["context"])
    for row in val_rows
)

print(f"\nUnique train contexts      : {len(train_contexts):,}")
print(f"Unique validation contexts: {len(val_contexts):,}")


# ---------------------------------------------------------
# Check overlap
# ---------------------------------------------------------

overlap = train_contexts.intersection(val_contexts)

print("\n" + "=" * 70)
print("DATASET OVERLAP")
print("=" * 70)

print(f"Contexts appearing in BOTH train and validation: {len(overlap):,}")

if len(val_contexts) > 0:
    percentage = len(overlap) / len(val_contexts) * 100
else:
    percentage = 0

print(f"Validation contexts found in train: {percentage:.2f}%")


# ---------------------------------------------------------
# Load BM25 index
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING BM25 INDEX")
print("=" * 70)

with open(INDEX_FILE, "rb") as f:
    index_data = pickle.load(f)


contexts = index_data.get("contexts", [])

print(f"Contexts stored in BM25 index: {len(contexts):,}")


# ---------------------------------------------------------
# Compare validation contexts against actual index
# ---------------------------------------------------------

index_contexts = set(
    normalize(context)
    for context in contexts
)

val_in_index = val_contexts.intersection(index_contexts)

print(
    f"Validation contexts found in BM25 index: "
    f"{len(val_in_index):,}"
)

if len(val_contexts) > 0:
    percentage_index = len(val_in_index) / len(val_contexts) * 100
else:
    percentage_index = 0

print(
    f"Percentage of validation contexts in index: "
    f"{percentage_index:.2f}%"
)


# ---------------------------------------------------------
# Final diagnosis
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if len(val_in_index) == 0:

    print(
        """
Your current BM25 Recall@K = 0 is NOT evidence that BM25
cannot retrieve the correct context.

The validation contexts are not present in the BM25 corpus.

Therefore, when the evaluator asks:

    "Did BM25 retrieve the gold validation context?"

the answer must always be NO.

This makes Recall@1/5/10 = 0 by experimental design.

We need to fix the evaluation protocol before comparing
BM25, MiniLM, RNN and Hybrid.
"""
    )

elif percentage_index < 50:

    print(
        f"""
Only {percentage_index:.2f}% of validation contexts exist
in the current BM25 corpus.

This means the current Recall@K calculation is still heavily
affected by corpus mismatch.

We should create a proper evaluation corpus.
"""
    )

else:

    print(
        f"""
{percentage_index:.2f}% of validation contexts are present
in the BM25 corpus.

The zero Recall@K result may therefore involve another
evaluation/mapping problem.

We will inspect the context IDs and retrieval mapping next.
"""
    )


print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
