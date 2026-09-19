import json
import random
import re
import time
import sys
import argparse
from pathlib import Path

import torch

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rnn_ranker import GRURanker


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = PROJECT_ROOT / "data" / "squad_train.jsonl"
MODEL_FILE = PROJECT_ROOT / "data" / "rnn_model.pt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "rnn_recall_evaluation.csv"


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Normal evaluation size
EVAL_SIZE = 2000

# Small test evaluation size
TEST_SIZE = 20

# Larger batch reduces the number of GRU forward passes.
# CPU memory permitting, 128 is substantially faster than 32.
BATCH_SIZE = 128

# Must match the trained model's Vocabulary.encode()
MAX_LEN = 192

PAD_ID = 0
UNK_ID = 1


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(
        r"[A-Za-z0-9]+",
        str(text).lower()
    )


# ============================================================
# NORMALIZATION
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

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_tokens(tokens, stoi):
    """
    Convert an already-tokenized list into vocabulary IDs.

    This avoids repeatedly running the regular-expression
    tokenizer inside the innermost evaluation loop.
    """

    ids = [
        stoi.get(token, UNK_ID)
        for token in tokens[:MAX_LEN]
    ]

    if not ids:
        ids = [UNK_ID]

    return ids


# ============================================================
# CREATE PADDED BATCH
# ============================================================

def create_batch(sequences):
    """
    Create a padded tensor and corresponding sequence lengths.
    """

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    ids = torch.zeros(
        (
            len(sequences),
            max_length
        ),
        dtype=torch.long
    )

    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long
        )

    return ids, lengths


# ============================================================
# PREPARE CONTEXTS
# ============================================================

def prepare_contexts(contexts, stoi):
    """
    Tokenize contexts once.

    The previous implementation tokenized the same contexts
    again for every question. This function removes that
    repeated work.
    """

    prepared = []

    for context in contexts:

        tokens = tokenize(context)

        # Keep only the part that can fit into MAX_LEN.
        tokens = tokens[:MAX_LEN]

        ids = encode_tokens(
            tokens,
            stoi
        )

        prepared.append(ids)

    return prepared


# ============================================================
# SCORE ONE QUESTION AGAINST ALL CONTEXTS
# ============================================================

def score_question(
    model,
    question,
    prepared_contexts,
    stoi
):
    """
    Score the question against every evaluation context.

    The trained GRU receives exactly the same structure used
    during training:

        question + [SEP] + context

    Because the model input has a fixed MAX_LEN, we preserve
    the original evaluator's truncation behavior.
    """

    question_tokens = tokenize(question)

    # The previous implementation created the complete
    # question + [SEP] + context string for every context.

    question_ids = [
        stoi.get(token, UNK_ID)
        for token in question_tokens
    ]

    separator_id = stoi.get("sep", UNK_ID)

    all_sequences = []

    for context_ids in prepared_contexts:

        combined = question_ids + [separator_id] + context_ids

        combined = combined[:MAX_LEN]

        if not combined:
            combined = [UNK_ID]

        all_sequences.append(combined)

    best_score = -float("inf")
    best_id = None

    # Process all contexts in larger batches.
    for start_index in range(
        0,
        len(all_sequences),
        BATCH_SIZE
    ):

        end_index = min(
            start_index + BATCH_SIZE,
            len(all_sequences)
        )

        batch_sequences = all_sequences[
            start_index:end_index
        ]

        ids, lengths = create_batch(
            batch_sequences
        )

        with torch.no_grad():
            scores = model(
                ids,
                lengths
            )

        scores = scores.cpu()

        local_best_index = int(
            torch.argmax(scores).item()
        )

        local_best_score = float(
            scores[local_best_index].item()
        )

        if local_best_score > best_score:

            best_score = local_best_score

            best_id = (
                start_index
                + local_best_index
            )

    return best_id


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a small 20-question test instead of 2,000 questions."
    )

    args = parser.parse_args()

    if args.test:
        evaluation_size = TEST_SIZE
        test_mode = True
    else:
        evaluation_size = EVAL_SIZE
        test_mode = False

    print("=" * 70)
    print("RNN / GRU RECALL EVALUATION")
    print("=" * 70)

    if test_mode:
        print("\n*** FAST TEST MODE: 20 QUESTIONS ***")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading SQuAD...")

    rows = load_jsonl(
        TRAIN_FILE
    )

    print(
        f"Total training examples: "
        f"{len(rows):,}"
    )

    # ========================================================
    # EXCLUDE GRU TRAINING DATA
    # ========================================================

    training_indices = set(
        range(
            min(
                20000,
                len(rows)
            )
        )
    )

    remaining_indices = [
        i
        for i in range(len(rows))
        if i not in training_indices
    ]

    # Reproduce the same random selection
    # used by the previous evaluator.
    random.seed(SEED)

    evaluation_indices = random.sample(
        remaining_indices,
        min(
            evaluation_size,
            len(remaining_indices)
        )
    )

    eval_rows = [
        rows[i]
        for i in evaluation_indices
    ]

    print(
        f"GRU training examples excluded: "
        f"{len(training_indices):,}"
    )

    print(
        f"Evaluation questions: "
        f"{len(eval_rows):,}"
    )

    # ========================================================
    # BUILD EVALUATION CORPUS
    # ========================================================

    contexts = []
    context_to_id = {}

    for row in eval_rows:

        context = row["context"]

        key = normalize(
            context
        )

        if key not in context_to_id:

            context_to_id[key] = len(
                contexts
            )

            contexts.append(
                context
            )

    print(
        f"Unique evaluation contexts: "
        f"{len(contexts):,}"
    )

    # ========================================================
    # QUESTIONS + GOLD CONTEXT IDs
    # ========================================================

    questions = []
    gold_ids = []

    for row in eval_rows:

        question = row["question"]
        context = row["context"]

        key = normalize(
            context
        )

        questions.append(
            question
        )

        gold_ids.append(
            context_to_id[key]
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print(
        "\nLoading trained GRU model..."
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )

    # ========================================================
    # LOAD VOCABULARY
    # ========================================================

    if "itos" in checkpoint:

        itos = checkpoint["itos"]

    elif "vocab" in checkpoint:

        vocab_data = checkpoint["vocab"]

        if isinstance(
            vocab_data,
            dict
        ):

            itos = vocab_data["itos"]

        else:

            itos = vocab_data.itos

    else:

        raise RuntimeError(
            "Vocabulary not found in rnn_model.pt"
        )

    stoi = {
        token: index
        for index, token in enumerate(
            itos
        )
    }

    print(
        f"Vocabulary size: "
        f"{len(itos):,}"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = GRURanker(
        vocab_size=len(itos),
        emb_dim=128,
        hidden_dim=160
    )

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

    else:

        raise RuntimeError(
            "Model weights not found in checkpoint."
        )

    model.eval()

    print(
        "GRU model loaded successfully."
    )

    # ========================================================
    # PREPARE CONTEXTS ONCE
    # ========================================================

    print(
        "\nPreparing evaluation contexts..."
    )

    preparation_start = time.perf_counter()

    prepared_contexts = prepare_contexts(
        contexts,
        stoi
    )

    preparation_time = (
        time.perf_counter()
        - preparation_start
    )

    print(
        f"Context preparation completed in "
        f"{preparation_time:.3f} seconds"
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    print(
        "\nEvaluating GRU retrieval..."
    )

    correct = 0
    total_latency = 0.0

    total_questions = len(
        questions
    )

    for question_number, question in enumerate(
        questions
    ):

        start_time = time.perf_counter()

        predicted_id = score_question(
            model,
            question,
            prepared_contexts,
            stoi
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        total_latency += latency

        # ----------------------------------------------------
        # Correct retrieval
        # ----------------------------------------------------

        if predicted_id == gold_ids[
            question_number
        ]:

            correct += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            question_number + 1
        ) % 5 == 0 or (
            question_number + 1
        ) == total_questions:

            print(
                f"Processed "
                f"{question_number + 1}/"
                f"{total_questions} "
                f"| Current latency: "
                f"{latency:.3f}s"
            )

    # ========================================================
    # CALCULATE RECALL
    # ========================================================

    total = total_questions

    recall = (
        correct / total
        if total > 0
        else 0.0
    )

    average_latency = (
        total_latency / total
        if total > 0
        else 0.0
    )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "model,correct,total,recall,average_latency\n"
        )

        f.write(
            f"RNN_GRU,"
            f"{correct},"
            f"{total},"
            f"{recall:.6f},"
            f"{average_latency:.6f}\n"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RNN / GRU RECALL RESULT"
    )

    print(
        "=" * 70
    )

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

    print(
        "\nResult saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 70
    )

    if test_mode:

        print(
            "\nFAST TEST COMPLETE."
        )

        print(
            "If this test completes successfully, "
            "we will decide whether the full evaluation "
            "needs another optimization before running it."
        )

    else:

        print(
            "\nRNN / GRU EVALUATION COMPLETE"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()