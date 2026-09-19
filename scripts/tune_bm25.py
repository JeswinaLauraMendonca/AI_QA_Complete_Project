from pathlib import Path
import json
import pickle
import re
import time
import csv

import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = ROOT / "data" / "index.pkl"
VALIDATION_FILE = ROOT / "data" / "squad_validation.jsonl"
RESULT_FILE = ROOT / "data" / "bm25_tuning_results.csv"


# ============================================================
# TUNING SETTINGS
# ============================================================

# Number of validation questions used for tuning.
# 1000 gives a good initial comparison.
LIMIT = 1000

# BM25 parameters to test.
K1_VALUES = [
    0.8,
    1.0,
    1.2,
    1.5,
    1.8,
    2.0
]

B_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0
]


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Same tokenizer used by the improved build_index.py.
    """

    text = str(text).lower()

    # Keep words and numbers.
    tokens = re.findall(
        r"\b[a-z0-9]+\b",
        text
    )

    return tokens


# ============================================================
# READ JSONL
# ============================================================

def read_jsonl(path):

    rows = []

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# SQUAD NORMALIZATION
# ============================================================

def normalize_answer(text):
    """
    SQuAD-style answer normalization.
    """

    text = str(text).lower()

    # Remove punctuation.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Normalize whitespace.
    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(
    prediction,
    ground_truth
):

    return int(
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


# ============================================================
# TOKEN F1
# ============================================================

def token_f1(
    prediction,
    ground_truth
):

    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    ground_truth_tokens = (
        normalize_answer(
            ground_truth
        ).split()
    )

    if (
        not prediction_tokens
        or
        not ground_truth_tokens
    ):
        return 0.0

    common = (
        set(prediction_tokens)
        &
        set(ground_truth_tokens)
    )

    common_count = sum(
        min(
            prediction_tokens.count(token),
            ground_truth_tokens.count(token)
        )
        for token in common
    )

    if common_count == 0:
        return 0.0

    precision = (
        common_count
        /
        len(prediction_tokens)
    )

    recall = (
        common_count
        /
        len(ground_truth_tokens)
    )

    return (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
    )


# ============================================================
# CHECK WHETHER ANSWER EXISTS IN CONTEXT
# ============================================================

def context_contains_answer(
    context,
    gold_answers
):
    """
    Returns one of the known SQuAD answers if the
    retrieved context contains it.
    Otherwise returns an empty string.
    """

    normalized_context = normalize_answer(
        context
    )

    for answer in gold_answers:

        normalized_answer = normalize_answer(
            answer
        )

        if (
            normalized_answer
            and
            normalized_answer in normalized_context
        ):

            return answer

    return ""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BM25 PARAMETER TUNING")
    print("=" * 80)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            f"\nIndex not found:\n{INDEX_FILE}\n\n"
            "Run:\n"
            "python scripts\\build_index.py"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"\nValidation dataset not found:\n"
            f"{VALIDATION_FILE}\n\n"
            "Run:\n"
            "python scripts\\download_dataset.py"
        )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    print("\nLoading index...")

    with open(
        INDEX_FILE,
        "rb"
    ) as f:

        index_data = pickle.load(f)

    contexts = list(
        index_data["contexts"]
    )

    print(
        f"Contexts loaded: {len(contexts):,}"
    )

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print("\nLoading validation data...")

    rows = read_jsonl(
        VALIDATION_FILE
    )

    rows = rows[:LIMIT]

    print(
        f"Validation questions used: {len(rows):,}"
    )

    # ========================================================
    # TOKENIZE CORPUS
    # ========================================================

    print("\nPreparing BM25 corpus...")

    tokenized_contexts = (
        index_data.get(
            "tokenized_contexts"
        )
    )

    if tokenized_contexts is None:

        tokenized_contexts = [
            tokenize(context)
            for context in contexts
        ]

    print(
        f"Tokenized contexts: "
        f"{len(tokenized_contexts):,}"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    total_configurations = (
        len(K1_VALUES)
        *
        len(B_VALUES)
    )

    configuration_number = 0

    # ========================================================
    # TEST EVERY BM25 CONFIGURATION
    # ========================================================

    for k1 in K1_VALUES:

        for b in B_VALUES:

            configuration_number += 1

            print("\n")
            print("-" * 80)

            print(
                f"Configuration "
                f"{configuration_number}"
                f"/"
                f"{total_configurations}"
            )

            print(
                f"k1 = {k1}"
            )

            print(
                f"b  = {b}"
            )

            # ------------------------------------------------
            # BUILD BM25
            # ------------------------------------------------

            build_start = (
                time.perf_counter()
            )

            bm25 = BM25Okapi(
                tokenized_contexts,
                k1=k1,
                b=b
            )

            build_time = (
                time.perf_counter()
                -
                build_start
            )

            # ------------------------------------------------
            # METRIC STORAGE
            # ------------------------------------------------

            em_scores = []
            f1_scores = []

            recall_at_1_count = 0
            recall_at_5_count = 0
            recall_at_10_count = 0

            query_times = []

            valid_examples = 0

            # =================================================
            # EVALUATE VALIDATION QUESTIONS
            # =================================================

            for number, row in enumerate(rows):

                question = str(
                    row.get(
                        "question",
                        ""
                    )
                ).strip()

                gold_context = str(
                    row.get(
                        "context",
                        ""
                    )
                )

                answers_data = row.get(
                    "answers",
                    {}
                )

                gold_answers = answers_data.get(
                    "text",
                    []
                )

                if (
                    not question
                    or
                    not gold_context
                    or
                    not gold_answers
                ):
                    continue

                valid_examples += 1

                # ------------------------------------------------
                # BM25 QUERY
                # ------------------------------------------------

                query_tokens = tokenize(
                    question
                )

                query_start = (
                    time.perf_counter()
                )

                scores = bm25.get_scores(
                    query_tokens
                )

                indices = np.argsort(
                    scores
                )[::-1]

                query_time = (
                    time.perf_counter()
                    -
                    query_start
                )

                query_times.append(
                    query_time
                )

                # ------------------------------------------------
                # RETRIEVED CONTEXTS
                # ------------------------------------------------

                top1_contexts = [
                    contexts[int(index)]
                    for index in indices[:1]
                ]

                top5_contexts = [
                    contexts[int(index)]
                    for index in indices[:5]
                ]

                top10_contexts = [
                    contexts[int(index)]
                    for index in indices[:10]
                ]

                # ------------------------------------------------
                # RETRIEVAL RECALL@1
                # ------------------------------------------------

                if gold_context in top1_contexts:

                    recall_at_1_count += 1

                # ------------------------------------------------
                # RETRIEVAL RECALL@5
                # ------------------------------------------------

                if gold_context in top5_contexts:

                    recall_at_5_count += 1

                # ------------------------------------------------
                # RETRIEVAL RECALL@10
                # ------------------------------------------------

                if gold_context in top10_contexts:

                    recall_at_10_count += 1

                # ------------------------------------------------
                # BEST RETRIEVED CONTEXT
                # ------------------------------------------------

                best_index = int(
                    indices[0]
                )

                predicted_context = (
                    contexts[best_index]
                )

                # ------------------------------------------------
                # FIND GOLD ANSWER IN RETRIEVED CONTEXT
                # ------------------------------------------------

                prediction = context_contains_answer(
                    predicted_context,
                    gold_answers
                )

                # ------------------------------------------------
                # ANSWER METRICS
                # ------------------------------------------------

                if prediction:

                    best_em = max(
                        exact_match(
                            prediction,
                            gold_answer
                        )
                        for gold_answer
                        in gold_answers
                    )

                    best_f1 = max(
                        token_f1(
                            prediction,
                            gold_answer
                        )
                        for gold_answer
                        in gold_answers
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

                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if (
                    number + 1
                ) % 100 == 0:

                    print(
                        f"Processed "
                        f"{number + 1}/"
                        f"{len(rows)}"
                    )

            # ====================================================
            # CALCULATE METRICS
            # ====================================================

            if valid_examples == 0:

                print(
                    "WARNING: No valid examples."
                )

                continue

            em = (
                sum(em_scores)
                /
                len(em_scores)
            )

            f1 = (
                sum(f1_scores)
                /
                len(f1_scores)
            )

            recall_at_1 = (
                recall_at_1_count
                /
                valid_examples
            )

            recall_at_5 = (
                recall_at_5_count
                /
                valid_examples
            )

            recall_at_10 = (
                recall_at_10_count
                /
                valid_examples
            )

            if query_times:

                latency = (
                    sum(query_times)
                    /
                    len(query_times)
                )

            else:

                latency = 0.0

            # ====================================================
            # SAVE RESULT
            # ====================================================

            result = {
                "k1": k1,
                "b": b,
                "EM": round(
                    em,
                    4
                ),
                "F1": round(
                    f1,
                    4
                ),
                "Recall@1": round(
                    recall_at_1,
                    4
                ),
                "Recall@5": round(
                    recall_at_5,
                    4
                ),
                "Recall@10": round(
                    recall_at_10,
                    4
                ),
                "Latency": round(
                    latency,
                    6
                ),
                "BuildTime": round(
                    build_time,
                    4
                )
            }

            results.append(
                result
            )

            # ====================================================
            # DISPLAY RESULT
            # ====================================================

            print("\nRESULT")

            print(
                f"EM        : "
                f"{em:.4f}"
            )

            print(
                f"F1        : "
                f"{f1:.4f}"
            )

            print(
                f"Recall@1  : "
                f"{recall_at_1:.4f}"
            )

            print(
                f"Recall@5  : "
                f"{recall_at_5:.4f}"
            )

            print(
                f"Recall@10 : "
                f"{recall_at_10:.4f}"
            )

            print(
                f"Latency   : "
                f"{latency:.6f} sec"
            )

    # ========================================================
    # MAKE SURE RESULTS EXIST
    # ========================================================

    if not results:

        raise RuntimeError(
            "No BM25 tuning results were generated."
        )

    # ========================================================
    # SORT
    #
    # Primary:
    # F1
    #
    # Secondary:
    # Recall@10
    #
    # Third:
    # EM
    # ========================================================

    results.sort(
        key=lambda item: (
            item["F1"],
            item["Recall@10"],
            item["EM"]
        ),
        reverse=True
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    print(
        "\nSaving tuning results..."
    )

    fieldnames = [
        "k1",
        "b",
        "EM",
        "F1",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Latency",
        "BuildTime"
    ]

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best = results[0]

    print("\n")
    print("=" * 80)
    print("BEST BM25 CONFIGURATION")
    print("=" * 80)

    print(
        f"k1        = {best['k1']}"
    )

    print(
        f"b         = {best['b']}"
    )

    print(
        f"EM        = {best['EM']:.4f}"
    )

    print(
        f"F1        = {best['F1']:.4f}"
    )

    print(
        f"Recall@1  = {best['Recall@1']:.4f}"
    )

    print(
        f"Recall@5  = {best['Recall@5']:.4f}"
    )

    print(
        f"Recall@10 = {best['Recall@10']:.4f}"
    )

    print(
        f"Latency   = {best['Latency']:.6f} sec"
    )

    print(
        f"\nAll results saved to:"
    )

    print(
        RESULT_FILE
    )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()