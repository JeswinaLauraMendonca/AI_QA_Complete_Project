from pathlib import Path
import json
import pickle
import re
import time
import csv

import numpy as np
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = ROOT / "data" / "index.pkl"
VALIDATION_FILE = ROOT / "data" / "squad_validation.jsonl"
RESULT_FILE = ROOT / "data" / "bm25_final_results.csv"

# Use the best candidate from tuning.
BM25_K1 = 0.8
BM25_B = 0.0

# Start with 1000 for speed.
LIMIT = 1000


def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def normalize_answer(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction, ground_truth):
    return int(
        normalize_answer(prediction)
        == normalize_answer(ground_truth)
    )


def token_f1(prediction, ground_truth):

    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()

    if not pred or not gold:
        return 0.0

    common = set(pred) & set(gold)

    common_count = sum(
        min(pred.count(token), gold.count(token))
        for token in common
    )

    if common_count == 0:
        return 0.0

    precision = common_count / len(pred)
    recall = common_count / len(gold)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def answer_in_context(context, answers):

    normalized_context = normalize_answer(context)

    for answer in answers:

        normalized_answer = normalize_answer(answer)

        if (
            normalized_answer
            and normalized_answer in normalized_context
        ):
            return answer

    return ""


def main():

    print("=" * 80)
    print("FINAL BM25 EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index not found:\n{INDEX_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{VALIDATION_FILE}"
        )

    # ---------------------------------------------------------
    # Load index
    # ---------------------------------------------------------

    print("\nLoading index...")

    with open(INDEX_FILE, "rb") as f:
        index_data = pickle.load(f)

    contexts = list(index_data["contexts"])

    print(
        f"Contexts: {len(contexts):,}"
    )

    # ---------------------------------------------------------
    # Tokenized corpus
    # ---------------------------------------------------------

    tokenized_contexts = index_data.get(
        "tokenized_contexts"
    )

    if tokenized_contexts is None:

        print("Creating tokens...")

        tokenized_contexts = [
            tokenize(context)
            for context in contexts
        ]

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    print("\nLoading validation dataset...")

    rows = read_jsonl(
        VALIDATION_FILE
    )[:LIMIT]

    print(
        f"Questions: {len(rows):,}"
    )

    # ---------------------------------------------------------
    # Build BM25
    # ---------------------------------------------------------

    print("\nBuilding final BM25...")

    bm25 = BM25Okapi(
        tokenized_contexts,
        k1=BM25_K1,
        b=BM25_B
    )

    print(
        f"k1 = {BM25_K1}"
    )

    print(
        f"b  = {BM25_B}"
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    em_scores = []
    f1_scores = []

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    latencies = []

    valid_count = 0

    # =========================================================
    # EVALUATION
    # =========================================================

    for i, row in enumerate(rows):

        question = str(
            row.get("question", "")
        ).strip()

        gold_context = str(
            row.get("context", "")
        )

        gold_answers = (
            row.get("answers", {})
            .get("text", [])
        )

        if (
            not question
            or not gold_context
            or not gold_answers
        ):
            continue

        valid_count += 1

        # -----------------------------------------------------
        # BM25 retrieval
        # -----------------------------------------------------

        query_tokens = tokenize(
            question
        )

        start = time.perf_counter()

        scores = bm25.get_scores(
            query_tokens
        )

        indices = np.argsort(
            scores
        )[::-1]

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # Compare the actual retrieved context text with
        # the SQuAD gold context.
        # -----------------------------------------------------

        retrieved_1 = [
            contexts[int(x)]
            for x in indices[:1]
        ]

        retrieved_5 = [
            contexts[int(x)]
            for x in indices[:5]
        ]

        retrieved_10 = [
            contexts[int(x)]
            for x in indices[:10]
        ]

        # -----------------------------------------------------
        # Recall@K
        # -----------------------------------------------------

        if gold_context in retrieved_1:
            recall_1 += 1

        if gold_context in retrieved_5:
            recall_5 += 1

        if gold_context in retrieved_10:
            recall_10 += 1

        # -----------------------------------------------------
        # Top-1 context
        # -----------------------------------------------------

        best_index = int(
            indices[0]
        )

        best_context = contexts[
            best_index
        ]

        # -----------------------------------------------------
        # Answer availability
        # -----------------------------------------------------

        prediction = answer_in_context(
            best_context,
            gold_answers
        )

        # -----------------------------------------------------
        # EM / F1
        # -----------------------------------------------------

        if prediction:

            best_em = max(
                exact_match(
                    prediction,
                    answer
                )
                for answer in gold_answers
            )

            best_f1 = max(
                token_f1(
                    prediction,
                    answer
                )
                for answer in gold_answers
            )

        else:

            best_em = 0
            best_f1 = 0.0

        em_scores.append(
            best_em
        )

        f1_scores.append(
            best_f1
        )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (i + 1) % 100 == 0:

            print(
                f"Processed {i + 1}/{len(rows)}"
            )

    # =========================================================
    # FINAL METRICS
    # =========================================================

    if valid_count == 0:
        raise RuntimeError(
            "No valid validation examples."
        )

    em = (
        sum(em_scores)
        / len(em_scores)
    )

    f1 = (
        sum(f1_scores)
        / len(f1_scores)
    )

    recall_at_1 = (
        recall_1
        / valid_count
    )

    recall_at_5 = (
        recall_5
        / valid_count
    )

    recall_at_10 = (
        recall_10
        / valid_count
    )

    latency = (
        sum(latencies)
        / len(latencies)
    )

    # =========================================================
    # DISPLAY
    # =========================================================

    print("\n")
    print("=" * 80)
    print("FINAL BM25 RESULTS")
    print("=" * 80)

    print(
        f"BM25 k1       : {BM25_K1}"
    )

    print(
        f"BM25 b        : {BM25_B}"
    )

    print(
        f"Exact Match   : {em:.4f} "
        f"({em * 100:.2f}%)"
    )

    print(
        f"F1            : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Recall@1      : {recall_at_1:.4f} "
        f"({recall_at_1 * 100:.2f}%)"
    )

    print(
        f"Recall@5      : {recall_at_5:.4f} "
        f"({recall_at_5 * 100:.2f}%)"
    )

    print(
        f"Recall@10     : {recall_at_10:.4f} "
        f"({recall_at_10 * 100:.2f}%)"
    )

    print(
        f"Avg Latency   : {latency:.6f} sec"
    )

    print("=" * 80)

    # =========================================================
    # SAVE
    # =========================================================

    result = {
        "model": "BM25",
        "k1": BM25_K1,
        "b": BM25_B,
        "EM": round(em, 4),
        "F1": round(f1, 4),
        "Recall@1": round(recall_at_1, 4),
        "Recall@5": round(recall_at_5, 4),
        "Recall@10": round(recall_at_10, 4),
        "Latency": round(latency, 6)
    }

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        writer.writeheader()
        writer.writerow(result)

    print(
        f"\nSaved to:\n{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()