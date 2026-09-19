from pathlib import Path
import pickle
import re
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from models.rnn_ranker import GRURanker, Vocabulary


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

INDEX_PATH = DATA_DIR / "index.pkl"
RNN_MODEL_PATH = DATA_DIR / "rnn_model.pt"

# Fine-tuned SentenceTransformer model
MINILM_MODEL_PATH = DATA_DIR / "minilm_finetuned"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# IMPORTANT:
# The final application now uses the fine-tuned MiniLM model.
EMBEDDING_MODEL_NAME = str(MINILM_MODEL_PATH)

# Original model name retained for documentation/reference.
BASE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local extractive reader
READER_MODEL_NAME = "distilbert-base-uncased-distilled-squad"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(text):
    """
    Normalize text for answer comparison and general processing.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_score_array(scores):
    """
    Min-max normalize an array of scores.

    Returns values between 0 and 1.
    """
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score - min_score < 1e-12:
        return np.ones_like(scores) * 0.5

    return (scores - min_score) / (max_score - min_score)


# ============================================================
# QA ENGINE
# ============================================================

class QAEngine:

    def __init__(self):

        print("=" * 65)
        print("INITIALIZING AI QUESTION ANSWERING SYSTEM")
        print("=" * 65)

        print(f"Device: {DEVICE}")

        # ----------------------------------------------------
        # LOAD RETRIEVAL INDEX
        # ----------------------------------------------------

        print("\nLoading retrieval index...")

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Retrieval index not found:\n{INDEX_PATH}"
            )

        with open(INDEX_PATH, "rb") as file:
            index_data = pickle.load(file)

        # ----------------------------------------------------
        # Support existing index formats
        # ----------------------------------------------------

        if isinstance(index_data, dict):

            self.contexts = index_data.get("contexts", [])
            self.bm25 = index_data.get("bm25")

            # Existing saved embeddings
            self.embeddings = index_data.get("embeddings")

            # Existing BM25 parameters if present
            self.bm25_k1 = index_data.get("bm25_k1", 1.2)
            self.bm25_b = index_data.get("bm25_b", 0.75)

        else:
            raise ValueError(
                "Unsupported retrieval index format."
            )

        if not self.contexts:
            raise ValueError(
                "No contexts were found in the retrieval index."
            )

        print(f"Loaded {len(self.contexts):,} contexts.")

        if self.bm25 is None:
            raise ValueError(
                "BM25 index was not found in data/index.pkl."
            )

        print("BM25 ready.")

        # ----------------------------------------------------
        # LOAD MINILM
        # ----------------------------------------------------

        print("\nLoading fine-tuned All-MiniLM-L6-v2...")

        if not MINILM_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Fine-tuned MiniLM model was not found at:\n"
                f"{MINILM_MODEL_PATH}"
            )

        self.mini = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=DEVICE
        )

        print("Fine-tuned All-MiniLM-L6-v2 ready.")

        print(
            "Embedding dimension:",
            self.mini.get_sentence_embedding_dimension()
        )

        # ----------------------------------------------------
        # CHECK EXISTING EMBEDDINGS
        # ----------------------------------------------------

        if self.embeddings is not None:

            self.embeddings = np.asarray(
                self.embeddings,
                dtype=np.float32
            )

            print(
                "Stored embedding matrix shape:",
                self.embeddings.shape
            )

            expected_dimension = (
                self.mini.get_sentence_embedding_dimension()
            )

            if (
                self.embeddings.ndim != 2
                or self.embeddings.shape[1] != expected_dimension
            ):
                print(
                    "Warning: stored embeddings do not match "
                    "the fine-tuned MiniLM dimension."
                )

        # ----------------------------------------------------
        # LOAD RNN / GRU
        # ----------------------------------------------------

        print("\nLoading RNN / GRU model...")

        self._load_rnn()

        print("RNN / GRU model ready.")

        # ----------------------------------------------------
        # LOAD EXTRACTIVE READER
        # ----------------------------------------------------

        print("\nLoading extractive answer reader...")

        self.reader_tokenizer = AutoTokenizer.from_pretrained(
            READER_MODEL_NAME
        )

        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(
            READER_MODEL_NAME
        )

        self.reader_model.to(DEVICE)
        self.reader_model.eval()

        print("Answer reader ready.")

        print("\nAll components loaded successfully.")
        print("=" * 65)

    # ========================================================
    # RNN LOADER
    # ========================================================

    def _load_rnn(self):

        if not RNN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"RNN model not found:\n{RNN_MODEL_PATH}"
            )

        checkpoint = torch.load(
            RNN_MODEL_PATH,
            map_location=DEVICE
        )

        # ----------------------------------------------------
        # Load vocabulary
        # ----------------------------------------------------

        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]

        self.rnn_vocab = Vocabulary()

        self.rnn_vocab.stoi = stoi
        self.rnn_vocab.itos = itos

        # ----------------------------------------------------
        # Model dimensions
        # ----------------------------------------------------

        emb_dim = checkpoint.get(
            "emb_dim",
            128
        )

        hidden_dim = checkpoint.get(
            "hidden_dim",
            160
        )

        # ----------------------------------------------------
        # Create GRU model
        # ----------------------------------------------------

        self.rnn = GRURanker(
            vocab_size=len(self.rnn_vocab.stoi),
            emb_dim=emb_dim,
            hidden_dim=hidden_dim
        )

        self.rnn.load_state_dict(
            checkpoint["state_dict"]
        )

        self.rnn.to(DEVICE)
        self.rnn.eval()

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(self, question, k=30):

        scores = np.asarray(
            self.bm25.get_scores(
                question.lower().split()
            ),
            dtype=np.float32
        )

        k = min(k, len(scores))

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        results = []

        for idx in top_indices:

            results.append({
                "index": int(idx),
                "context": self.contexts[int(idx)],
                "score": float(scores[int(idx)])
            })

        return results

    # ========================================================
    # MINILM SEARCH
    # ========================================================

    def minilm_search(self, question, k=30):

        # ----------------------------------------------------
        # If stored embeddings are available and compatible,
        # use them for document retrieval.
        # ----------------------------------------------------

        if (
            self.embeddings is not None
            and self.embeddings.ndim == 2
            and self.embeddings.shape[0] == len(self.contexts)
            and self.embeddings.shape[1]
            == self.mini.get_sentence_embedding_dimension()
        ):

            query_embedding = self.mini.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]

            scores = np.dot(
                self.embeddings,
                query_embedding
            )

        else:

            # ------------------------------------------------
            # Safety fallback:
            # create embeddings using the same fine-tuned
            # model currently loaded by the application.
            # ------------------------------------------------

            document_embeddings = self.mini.encode(
                self.contexts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            query_embedding = self.mini.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]

            scores = np.dot(
                document_embeddings,
                query_embedding
            )

        scores = np.asarray(
            scores,
            dtype=np.float32
        )

        k = min(k, len(scores))

        top_indices = np.argsort(
            scores
        )[::-1][:k]

        results = []

        for idx in top_indices:

            results.append({
                "index": int(idx),
                "context": self.contexts[int(idx)],
                "score": float(scores[int(idx)])
            })

        return results

    # ========================================================
    # RNN / GRU SCORING
    # ========================================================

    def _rnn_score(self, question, context):

        text = (
            str(question)
            + " [SEP] "
            + str(context)
        )

        ids = self.rnn_vocab.encode(
            text,
            max_len=192
        )

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device=DEVICE
        )

        lengths = torch.tensor(
            [len(ids)],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            output = self.rnn(
                input_ids,
                lengths
            )

            probability = torch.sigmoid(
                output
            ).item()

        return float(probability)

    # ========================================================
    # RNN SEARCH
    # ========================================================

    def rnn_search(self, question, k=30):

        # ----------------------------------------------------
        # RNN is used as a reranker over BM25 candidates.
        # ----------------------------------------------------

        candidate_k = max(
            40,
            k
        )

        candidates = self.bm25_search(
            question,
            k=candidate_k
        )

        scored = []

        for item in candidates:

            rnn_score = self._rnn_score(
                question,
                item["context"]
            )

            scored.append({
                "index": item["index"],
                "context": item["context"],
                "score": rnn_score
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return scored[:k]

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(self, question, k=30):

        # ----------------------------------------------------
        # Retrieve candidates from all approaches.
        # ----------------------------------------------------

        bm25_results = self.bm25_search(
            question,
            k=30
        )

        minilm_results = self.minilm_search(
            question,
            k=30
        )

        rnn_results = self.rnn_search(
            question,
            k=30
        )

        # ----------------------------------------------------
        # Create lookup dictionaries
        # ----------------------------------------------------

        bm25_dict = {
            item["index"]: item["score"]
            for item in bm25_results
        }

        minilm_dict = {
            item["index"]: item["score"]
            for item in minilm_results
        }

        rnn_dict = {
            item["index"]: item["score"]
            for item in rnn_results
        }

        # ----------------------------------------------------
        # Union of candidates
        # ----------------------------------------------------

        candidate_indices = set()

        candidate_indices.update(
            bm25_dict.keys()
        )

        candidate_indices.update(
            minilm_dict.keys()
        )

        candidate_indices.update(
            rnn_dict.keys()
        )

        # ----------------------------------------------------
        # Normalize individual scores
        # ----------------------------------------------------

        bm25_values = np.array(
            list(bm25_dict.values()),
            dtype=np.float32
        )

        minilm_values = np.array(
            list(minilm_dict.values()),
            dtype=np.float32
        )

        rnn_values = np.array(
            list(rnn_dict.values()),
            dtype=np.float32
        )

        bm25_norm_values = normalize_score_array(
            bm25_values
        )

        minilm_norm_values = normalize_score_array(
            minilm_values
        )

        rnn_norm_values = normalize_score_array(
            rnn_values
        )

        bm25_norm = dict(
            zip(
                bm25_dict.keys(),
                bm25_norm_values
            )
        )

        minilm_norm = dict(
            zip(
                minilm_dict.keys(),
                minilm_norm_values
            )
        )

        rnn_norm = dict(
            zip(
                rnn_dict.keys(),
                rnn_norm_values
            )
        )

        # ----------------------------------------------------
        # Fixed hybrid weights
        # ----------------------------------------------------

        BM25_WEIGHT = 0.45
        MINILM_WEIGHT = 0.40
        RNN_WEIGHT = 0.15

        # ----------------------------------------------------
        # Fuse scores
        # ----------------------------------------------------

        fused_results = []

        for idx in candidate_indices:

            score = (
                BM25_WEIGHT
                * bm25_norm.get(idx, 0.0)
                +
                MINILM_WEIGHT
                * minilm_norm.get(idx, 0.0)
                +
                RNN_WEIGHT
                * rnn_norm.get(idx, 0.0)
            )

            fused_results.append({
                "index": int(idx),
                "context": self.contexts[int(idx)],
                "score": float(score)
            })

        fused_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fused_results[:k]

    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    def extract_answer(self, question, context):

        if not context:
            return ""

        inputs = self.reader_tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop(
            "offset_mapping"
        )

        # ----------------------------------------------------
        # Determine which tokens belong to context.
        # ----------------------------------------------------

        sequence_ids = inputs.sequence_ids()

        context_token_indices = [
            i
            for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
        ]

        if not context_token_indices:
            return ""

        context_start = context_token_indices[0]
        context_end = context_token_indices[-1]

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.reader_model(
                **inputs
            )

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # ----------------------------------------------------
        # Best answer span
        # ----------------------------------------------------

        best_score = -float("inf")

        best_start = context_start
        best_end = context_start

        max_answer_length = 30

        for start in range(
            context_start,
            context_end + 1
        ):

            for end in range(
                start,
                min(
                    start + max_answer_length,
                    context_end + 1
                )
            ):

                score = (
                    start_logits[start].item()
                    +
                    end_logits[end].item()
                )

                if score > best_score:

                    best_score = score
                    best_start = start
                    best_end = end

        # ----------------------------------------------------
        # Convert token span to text span
        # ----------------------------------------------------

        start_offset = offset_mapping[
            0, best_start
        ].tolist()

        end_offset = offset_mapping[
            0, best_end
        ].tolist()

        answer = context[
            start_offset[0]:end_offset[1]
        ]

        return answer.strip()

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        question,
        method="hybrid",
        top_k=5
    ):

        if not question or not question.strip():

            return {
                "question": question,
                "method": method,
                "answer": "",
                "results": [],
                "latency": 0.0
            }

        question = question.strip()

        start_time = time.perf_counter()

        method = method.lower()

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        if method == "bm25":

            retrieved = self.bm25_search(
                question,
                k=max(top_k, 30)
            )

        elif method == "minilm":

            retrieved = self.minilm_search(
                question,
                k=max(top_k, 30)
            )

        elif method == "rnn":

            retrieved = self.rnn_search(
                question,
                k=max(top_k, 30)
            )

        elif method == "hybrid":

            retrieved = self.hybrid_search(
                question,
                k=max(top_k, 30)
            )

        else:

            raise ValueError(
                "Unknown method. Choose one of: "
                "bm25, minilm, rnn, hybrid."
            )

        # ----------------------------------------------------
        # Keep requested number of results
        # ----------------------------------------------------

        retrieved = retrieved[:top_k]

        # ----------------------------------------------------
        # Extract answer from top retrieved context
        # ----------------------------------------------------

        if retrieved:

            best_context = retrieved[0]["context"]

            answer = self.extract_answer(
                question,
                best_context
            )

        else:

            answer = ""

        latency = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Prepare results
        # ----------------------------------------------------

        results = []

        for item in retrieved:

            results.append({
                "context": item["context"],
                "score": item["score"],
                "index": item["index"]
            })

        return {
            "question": question,
            "method": method,
            "answer": answer,
            "results": results,
            "latency": latency
        }

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question,
        method="hybrid"
    ):

        return self.query(
            question=question,
            method=method,
            top_k=5
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    engine = QAEngine()

    test_question = (
        "What is the capital of Pakistan?"
    )

    print("\nTest Question:")
    print(test_question)

    for method in [
        "bm25",
        "minilm",
        "rnn",
        "hybrid"
    ]:

        print("\n" + "-" * 65)
        print(f"METHOD: {method.upper()}")
        print("-" * 65)

        result = engine.ask(
            test_question,
            method=method
        )

        print(
            "Answer:",
            result["answer"]
        )

        print(
            "Latency:",
            f"{result['latency']:.4f} seconds"
        )

        if result["results"]:

            print(
                "Top retrieval score:",
                result["results"][0]["score"]
            )