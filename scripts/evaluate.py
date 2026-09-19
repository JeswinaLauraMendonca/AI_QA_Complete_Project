from pathlib import Path
import csv
import json
import random
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.qa_engine import QAEngine


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION = ROOT / "data" / "squad_validation.jsonl"

DEV_FILE = ROOT / "data" / "evaluation_dev.jsonl"
TEST_FILE = ROOT / "data" / "evaluation_test.jsonl"

RESULTS = ROOT / "data" / "evaluation_results.csv"

SEED = 42

DEV_SIZE = 1000
TEST_SIZE = 1000

METHODS = [
    "bm25",
    "minilm",
    "rnn",
    "hybrid",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    """
    SQuAD-style normalization.
    """

    text = str(text).lower()

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def tokens(text):
    return normalize(text).split()


def exact_match(prediction, gold):
    return int(
        normalize(prediction)
        ==
        normalize(gold)
    )


def token_prf(prediction, gold):
    """
    Calculate token-level precision,
    recall and F1.
    """

    pred_tokens = tokens(prediction)
    gold_tokens = tokens(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = {}
    gold_counts = {}

    for token in pred_tokens:
        pred_counts[token] = (
            pred_counts.get(token, 0) + 1
        )

    for token in gold_tokens:
        gold_counts[token] = (
            gold_counts.get(token, 0) + 1
        )

    overlap = 0

    for token, count in gold_counts.items():

        overlap += min(
            pred_counts.get(token, 0),
            count
        )

    precision = (
        overlap / len(pred_tokens)
    )

    recall = (
        overlap / len(gold_tokens)
    )

    if precision + recall == 0:

        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    return precision, recall, f1


# ============================================================
# LOAD SQUAD VALIDATION DATA
# ============================================================

def load_squad_jsonl(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nValidation dataset not found:\n"
            f"{path}\n\n"
            f"Run:\n"
            f"python scripts/download_dataset.py"
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


# ============================================================
# CREATE FIXED DEVELOPMENT / TEST SPLIT
# ============================================================

def create_fixed_split(rows):

    """
    Create a reproducible split.

    Seed = 42

    Development set:
        1,000 questions

    Final test set:
        1,000 questions
    """

    rng = random.Random(SEED)

    indices = list(
        range(len(rows))
    )

    rng.shuffle(indices)

    required = (
        DEV_SIZE
        +
        TEST_SIZE
    )

    if len(indices) < required:

        raise ValueError(
            f"Not enough validation questions.\n"
            f"Available: {len(indices)}\n"
            f"Required: {required}"
        )

    dev_indices = indices[
        :DEV_SIZE
    ]

    test_indices = indices[
        DEV_SIZE:
        DEV_SIZE + TEST_SIZE
    ]

    dev_rows = [
        rows[i]
        for i in dev_indices
    ]

    test_rows = [
        rows[i]
        for i in test_indices
    ]

    # Save development questions

    with open(
        DEV_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in dev_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                +
                "\n"
            )

    # Save final test questions

    with open(
        TEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in test_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                +
                "\n"
            )

    return dev_rows, test_rows


# ============================================================
# BUILD COMMON VALIDATION CORPUS
# ============================================================

def build_validation_corpus(rows):

    """
    Build one common corpus from all unique
    validation contexts.

    All four approaches use the same corpus.
    """

    contexts = []
    seen = set()

    for row in rows:

        context = row.get(
            "context",
            ""
        ).strip()

        if not context:
            continue

        if context not in seen:

            seen.add(context)

            contexts.append(context)

    return contexts


def tokenize_for_bm25(text):

    return normalize(text).split()


# ============================================================
# FAIR EVALUATION ENGINE
# ============================================================

class FairEvaluationEngine:

    def __init__(self, contexts):

        print(
            "\nLoading existing QAEngine..."
        )

        # Reuse all existing project models.
        self.engine = QAEngine()

        self.contexts = contexts

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        print(
            f"Building BM25 index over "
            f"{len(contexts):,} validation contexts..."
        )

        tokenized_contexts = [
            tokenize_for_bm25(context)
            for context in contexts
        ]

        # Frozen BM25 parameters from
        # the existing project tuning.
        self.bm25 = BM25Okapi(
            tokenized_contexts,
            k1=0.8,
            b=1.0
        )

        # ----------------------------------------------------
        # MiniLM
        # ----------------------------------------------------

        print(
            "Building MiniLM embeddings "
            "for validation corpus..."
        )

        # IMPORTANT:
        #
        # QAEngine:
        #     self.mini       = SentenceTransformer model
        #     self.embeddings = NumPy context matrix
        #
        # Therefore .encode() must be called
        # on self.engine.mini.

        validation_embeddings = (
            self.engine.mini.encode(
                contexts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        self.validation_embeddings = (
            validation_embeddings.astype(
                np.float32
            )
        )

        print(
            "Validation retrieval corpus ready."
        )

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        question,
        k=30
    ):

        query_tokens = tokenize_for_bm25(
            question
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        k = min(
            k,
            len(scores)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            {
                "index": int(idx),
                "context": self.contexts[
                    int(idx)
                ],
                "score": float(
                    scores[int(idx)]
                )
            }
            for idx in top_indices
        ]

    # ========================================================
    # MINILM SEARCH
    # ========================================================

    def minilm_search(
        self,
        question,
        k=30
    ):

        # Use the actual SentenceTransformer
        # model stored as self.engine.mini.

        query_embedding = (
            self.engine.mini.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
        )

        scores = (
            self.validation_embeddings
            @
            query_embedding
        )

        k = min(
            k,
            len(scores)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            {
                "index": int(idx),
                "context": self.contexts[
                    int(idx)
                ],
                "score": float(
                    scores[int(idx)]
                )
            }
            for idx in top_indices
        ]

    # ========================================================
    # RNN / GRU SEARCH
    # ========================================================

    def rnn_search(
        self,
        question,
        k=40
    ):

        """
        Existing project architecture:

        Question
             |
             v
        BM25 candidates
             |
             v
        GRU reranking
             |
             v
        Ranked contexts
        """

        candidates = self.bm25_search(
            question,
            k=40
        )

        scored = []

        for item in candidates:

            score = self.engine._rnn_score(
                question,
                item["context"]
            )

            scored.append(
                {
                    "index": item["index"],
                    "context": item["context"],
                    "score": float(score)
                }
            )

        scored.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return scored[:k]

    # ========================================================
    # SCORE NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_scores(items):

        if not items:
            return {}

        values = np.array(
            [
                item["score"]
                for item in items
            ],
            dtype=np.float32
        )

        minimum = float(
            values.min()
        )

        maximum = float(
            values.max()
        )

        if (
            maximum - minimum
            <
            1e-12
        ):

            return {
                item["index"]: 1.0
                for item in items
            }

        return {
            item["index"]:
            float(
                (
                    item["score"]
                    - minimum
                )
                /
                (
                    maximum
                    -
                    minimum
                )
            )
            for item in items
        }

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        question,
        k=30
    ):

        bm25_results = (
            self.bm25_search(
                question,
                k=30
            )
        )

        minilm_results = (
            self.minilm_search(
                question,
                k=30
            )
        )

        rnn_results = (
            self.rnn_search(
                question,
                k=30
            )
        )

        bm25_scores = (
            self.normalize_scores(
                bm25_results
            )
        )

        minilm_scores = (
            self.normalize_scores(
                minilm_results
            )
        )

        rnn_scores = (
            self.normalize_scores(
                rnn_results
            )
        )

        candidate_indices = set()

        candidate_indices.update(
            bm25_scores.keys()
        )

        candidate_indices.update(
            minilm_scores.keys()
        )

        candidate_indices.update(
            rnn_scores.keys()
        )

        fused = []

        for idx in candidate_indices:

            final_score = (
                0.45
                *
                bm25_scores.get(
                    idx,
                    0.0
                )
                +
                0.40
                *
                minilm_scores.get(
                    idx,
                    0.0
                )
                +
                0.15
                *
                rnn_scores.get(
                    idx,
                    0.0
                )
            )

            fused.append(
                {
                    "index": idx,
                    "context": self.contexts[
                        idx
                    ],
                    "score": final_score
                }
            )

        fused.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fused[:k]

    # ========================================================
    # RETRIEVAL DISPATCH
    # ========================================================

    def retrieve(
        self,
        question,
        method
    ):

        if method == "bm25":

            return self.bm25_search(
                question,
                k=30
            )

        elif method == "minilm":

            return self.minilm_search(
                question,
                k=30
            )

        elif method == "rnn":

            return self.rnn_search(
                question,
                k=40
            )

        elif method == "hybrid":

            return self.hybrid_search(
                question,
                k=30
            )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

    # ========================================================
    # SHARED ANSWER READER
    # ========================================================

    def answer(
        self,
        question,
        method
    ):

        results = self.retrieve(
            question,
            method
        )

        if not results:
            return ""

        # Use the top-ranked context.
        top_context = results[0][
            "context"
        ]

        # Same extractive reader for
        # every retrieval method.
        prediction = (
            self.engine.extract_answer(
                question,
                top_context
            )
        )

        return prediction


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    evaluator,
    rows,
    method
):

    em_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0
    latency_total = 0.0

    print(
        f"\nEvaluating {method.upper()}..."
    )

    for number, row in enumerate(
        rows,
        start=1
    ):

        question = row[
            "question"
        ]

        gold_answers = row[
            "answers"
        ][
            "text"
        ]

        # --------------------------------------------
        # Start timing immediately before inference.
        # --------------------------------------------

        start = time.perf_counter()

        prediction = evaluator.answer(
            question,
            method
        )

        elapsed = (
            time.perf_counter()
            -
            start
        )

        latency_total += elapsed

        # --------------------------------------------
        # Exact Match
        # --------------------------------------------

        em_value = max(
            exact_match(
                prediction,
                gold
            )
            for gold in gold_answers
        )

        # --------------------------------------------
        # Precision / Recall / F1
        # --------------------------------------------

        metric_values = [
            token_prf(
                prediction,
                gold
            )
            for gold in gold_answers
        ]

        best_precision, best_recall, best_f1 = max(
            metric_values,
            key=lambda x: x[2]
        )

        em_total += em_value

        precision_total += (
            best_precision
        )

        recall_total += (
            best_recall
        )

        f1_total += (
            best_f1
        )

        # --------------------------------------------
        # Progress
        # --------------------------------------------

        if (
            number % 100 == 0
            or number == len(rows)
        ):

            print(
                f"  {number:,}/"
                f"{len(rows):,} "
                f"questions completed"
            )

    n = len(rows)

    return {
        "method": method,

        "exact_match":
            em_total / n,

        "precision":
            precision_total / n,

        "recall":
            recall_total / n,

        "f1":
            f1_total / n,

        "avg_latency_seconds":
            latency_total / n
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(summary):

    RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "method",
        "exact_match",
        "precision",
        "recall",
        "f1",
        "avg_latency_seconds"
    ]

    with open(
        RESULTS,
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
            summary
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "FAIR COMMON-CORPUS EVALUATION"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading SQuAD validation data..."
    )

    all_rows = load_squad_jsonl(
        VALIDATION
    )

    print(
        f"Validation questions available: "
        f"{len(all_rows):,}"
    )

    # ========================================================
    # FIXED SPLIT
    # ========================================================

    print(
        "\nCreating reproducible "
        "development/test split..."
    )

    dev_rows, test_rows = (
        create_fixed_split(
            all_rows
        )
    )

    print(
        f"Development questions: "
        f"{len(dev_rows):,}"
    )

    print(
        f"Final test questions:  "
        f"{len(test_rows):,}"
    )

    print(
        f"Random seed: {SEED}"
    )

    # ========================================================
    # COMMON CORPUS
    # ========================================================

    print(
        "\nBuilding common retrieval "
        "corpus from SQuAD validation "
        "contexts..."
    )

    corpus = (
        build_validation_corpus(
            all_rows
        )
    )

    print(
        f"Unique validation contexts: "
        f"{len(corpus):,}"
    )

    # ========================================================
    # INITIALIZE
    # ========================================================

    evaluator = (
        FairEvaluationEngine(
            corpus
        )
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        f"FINAL TEST EVALUATION "
        f"({len(test_rows):,} QUESTIONS)"
    )

    print(
        "=" * 90
    )

    summary = []

    for method in METHODS:

        result = evaluate_method(
            evaluator,
            test_rows,
            method
        )

        summary.append(
            result
        )

        print(
            f"\n{method.upper()} RESULTS"
        )

        print(
            f"  Exact Match : "
            f"{result['exact_match']:.4f}"
        )

        print(
            f"  Precision   : "
            f"{result['precision']:.4f}"
        )

        print(
            f"  Recall      : "
            f"{result['recall']:.4f}"
        )

        print(
            f"  F1 Score    : "
            f"{result['f1']:.4f}"
        )

        print(
            f"  Avg Latency : "
            f"{result['avg_latency_seconds']:.4f} s"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Method':<12}"
        f"{'EM':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Latency(s)':>14}"
    )

    print(
        "-" * 90
    )

    for result in summary:

        print(
            f"{result['method']:<12}"
            f"{result['exact_match']:>10.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['avg_latency_seconds']:>14.4f}"
        )

    print(
        "-" * 90
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        summary
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS
    )

    print(
        "\nDevelopment split saved to:"
    )

    print(
        DEV_FILE
    )

    print(
        "\nFinal test split saved to:"
    )

    print(
        TEST_FILE
    )

    # ========================================================
    # FINAL NOTES
    # ========================================================

    print(
        "\nIMPORTANT:"
    )

    print(
        "These numerical results are generated "
        "by YOUR implementation."
    )

    print(
        "Do not copy numerical results "
        "from the research papers."
    )

    print(
        "\nFinal evaluation metrics:"
    )

    print(
        "1. Exact Match"
    )

    print(
        "2. Precision"
    )

    print(
        "3. Recall"
    )

    print(
        "4. F1 Score"
    )

    print(
        "5. Average Response Latency"
    )

    print(
        "\nUser satisfaction is NOT included."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()