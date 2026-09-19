from pathlib import Path
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

TRAIN_FILE = DATA_DIR / "squad_train.jsonl"
INDEX_FILE = DATA_DIR / "index.pkl"

# Fine-tuned MiniLM model
MINILM_MODEL = DATA_DIR / "minilm_finetuned"


# ============================================================
# CONFIGURATION
# ============================================================

# Frozen BM25 parameters selected during development tuning.
BM25_K1 = 0.8
BM25_B = 1.0

# Embedding configuration
EMBEDDING_BATCH_SIZE = 32


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Simple lowercase whitespace/punctuation tokenizer
    for BM25.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().split()


# ============================================================
# LOAD SQUAD TRAINING DATA
# ============================================================

def load_contexts():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"SQuAD training file not found:\n{TRAIN_FILE}"
        )

    contexts = []
    seen = set()

    print("\nLoading SQuAD training contexts...")

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            import json

            item = json.loads(line)

            context = item.get(
                "context",
                ""
            )

            context = context.strip()

            if not context:
                continue

            # Keep only unique contexts
            if context not in seen:

                seen.add(context)
                contexts.append(context)

    print(
        f"Unique contexts loaded: {len(contexts):,}"
    )

    return contexts


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(contexts):

    print("\nBuilding BM25 index...")

    tokenized_contexts = [
        tokenize(context)
        for context in contexts
    ]

    bm25 = BM25Okapi(
        tokenized_contexts,
        k1=BM25_K1,
        b=BM25_B
    )

    print("BM25 index ready.")

    print(
        f"BM25 parameters: "
        f"k1={BM25_K1}, b={BM25_B}"
    )

    return bm25


# ============================================================
# BUILD MINILM EMBEDDINGS
# ============================================================

def build_embeddings(contexts):

    if not MINILM_MODEL.exists():
        raise FileNotFoundError(
            "Fine-tuned MiniLM model not found at:\n"
            f"{MINILM_MODEL}"
        )

    print(
        "\nLoading fine-tuned "
        "All-MiniLM-L6-v2..."
    )

    model = SentenceTransformer(
        str(MINILM_MODEL)
    )

    print(
        "Fine-tuned MiniLM loaded."
    )

    print(
        "Embedding dimension:",
        model.get_embedding_dimension()
    )

    print(
        f"\nGenerating embeddings for "
        f"{len(contexts):,} contexts..."
    )

    embeddings = model.encode(
        contexts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        "\nEmbedding matrix shape:",
        embeddings.shape
    )

    return embeddings


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    contexts,
    bm25,
    embeddings
):

    index_data = {

        "contexts": contexts,

        "bm25": bm25,

        "embeddings": embeddings,

        "bm25_k1": BM25_K1,

        "bm25_b": BM25_B,

        "embedding_model": (
            "data/minilm_finetuned"
        ),

        "embedding_dimension": int(
            embeddings.shape[1]
        )
    }

    with open(
        INDEX_FILE,
        "wb"
    ) as file:

        pickle.dump(
            index_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        f"\nIndex saved successfully:\n"
        f"{INDEX_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BUILDING FINAL TRAINING RETRIEVAL INDEX"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load contexts
    # --------------------------------------------------------

    contexts = load_contexts()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25 = build_bm25(
        contexts
    )

    # --------------------------------------------------------
    # Fine-tuned MiniLM
    # --------------------------------------------------------

    embeddings = build_embeddings(
        contexts
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(contexts) != len(embeddings):

        raise RuntimeError(
            "Number of contexts and embeddings "
            "does not match."
        )

    if embeddings.shape[1] != 384:

        raise RuntimeError(
            "Unexpected MiniLM embedding "
            f"dimension: {embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        contexts,
        bm25,
        embeddings
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nContexts: "
        f"{len(contexts):,}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"BM25 k1: "
        f"{BM25_K1}"
    )

    print(
        f"BM25 b: "
        f"{BM25_B}"
    )

    print(
        "\nEmbedding model:"
    )

    print(
        "data/minilm_finetuned"
    )

    print(
        "\nIndex:"
    )

    print(
        INDEX_FILE
    )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()