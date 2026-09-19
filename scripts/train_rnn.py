import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

from models.rnn_ranker import Vocabulary, GRURanker, make_pair


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = ROOT / "data" / "squad_train.jsonl"
MODEL_FILE = ROOT / "data" / "rnn_model.pt"

SEED = 42

# Increase this later if training is stable.
MAX_ROWS = 20000

EPOCHS = 5

BATCH_SIZE = 64

LEARNING_RATE = 0.0005

MAX_SEQUENCE_LENGTH = 192


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    print("ERROR: SQuAD training dataset not found.")

    print(
        "Run:\n"
        "python scripts\\download_dataset.py"
    )

    sys.exit(1)


# ============================================================
# HEADER
# ============================================================

print("=" * 65)

print("AI QUESTION ANSWERING SYSTEM")

print("IMPROVED RNN / GRU TRAINING")

print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading SQuAD training data...")

rows = []

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        rows.append(
            json.loads(line)
        )


print(
    f"Total training examples: {len(rows)}"
)


# ============================================================
# LIMIT DATASET
# ============================================================

rows = rows[:MAX_ROWS]

print(
    f"Examples used for training: {len(rows)}"
)


# ============================================================
# UNIQUE CONTEXTS
# ============================================================

contexts = list(
    dict.fromkeys(
        row["context"]
        for row in rows
    )
)

print(
    f"Unique contexts: {len(contexts)}"
)


# ============================================================
# BUILD VOCABULARY
# ============================================================

print("\nBuilding vocabulary...")

training_texts = []

for row in rows:

    training_texts.append(
        make_pair(
            row["question"],
            row["context"]
        )
    )


vocab = Vocabulary(
    min_freq=2,
    max_size=40000
)

vocab.build(
    training_texts
)

print(
    f"Vocabulary size: {len(vocab.itos)}"
)


# ============================================================
# DATASET
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        pairs,
        vocab
    ):

        self.pairs = pairs

        self.vocab = vocab


    def __len__(self):

        return len(self.pairs)


    def __getitem__(
        self,
        index
    ):

        text, label = self.pairs[index]

        sequence = self.vocab.encode(
            text,
            max_len=MAX_SEQUENCE_LENGTH
        )

        return (
            sequence,
            label
        )


# ============================================================
# CREATE POSITIVE + HARD NEGATIVE EXAMPLES
# ============================================================

print(
    "\nCreating training pairs..."
)

pairs = []

for row in rows:

    question = row["question"]

    positive_context = row["context"]

    # --------------------------------------------------------
    # Positive pair
    # --------------------------------------------------------

    positive_text = make_pair(
        question,
        positive_context
    )

    pairs.append(
        (
            positive_text,
            1.0
        )
    )

    # --------------------------------------------------------
    # Negative context
    #
    # Instead of completely random negatives only,
    # choose a context sharing some question words.
    # This creates harder training examples.
    # --------------------------------------------------------

    question_words = set(
        question.lower().split()
    )

    candidate_contexts = []

    for context in random.sample(
        contexts,
        min(30, len(contexts))
    ):

        if context == positive_context:
            continue

        context_words = set(
            context.lower().split()
        )

        overlap = len(
            question_words
            &
            context_words
        )

        candidate_contexts.append(
            (
                overlap,
                context
            )
        )

    if candidate_contexts:

        candidate_contexts.sort(
            key=lambda x: x[0],
            reverse=True
        )

        negative_context = (
            candidate_contexts[0][1]
        )

    else:

        negative_context = random.choice(
            contexts
        )

    negative_text = make_pair(
        question,
        negative_context
    )

    pairs.append(
        (
            negative_text,
            0.0
        )
    )


print(
    f"Total training pairs: {len(pairs)}"
)


# ============================================================
# COLLATE
# ============================================================

def collate_batch(batch):

    sequences, labels = zip(
        *batch
    )

    lengths = torch.tensor(
        [
            len(sequence)
            for sequence in sequences
        ],
        dtype=torch.long
    )

    max_length = int(
        lengths.max().item()
    )

    input_ids = torch.zeros(
        (
            len(sequences),
            max_length
        ),
        dtype=torch.long
    )

    for i, sequence in enumerate(
        sequences
    ):

        input_ids[
            i,
            :len(sequence)
        ] = torch.tensor(
            sequence,
            dtype=torch.long
        )

    labels = torch.tensor(
        labels,
        dtype=torch.float32
    )

    return (
        input_ids,
        lengths,
        labels
    )


# ============================================================
# DATALOADER
# ============================================================

dataset = PairDataset(
    pairs,
    vocab
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

    print("\nGPU detected.")

else:

    device = torch.device(
        "cpu"
    )

    print("\nGPU not detected.")

    print(
        "Training will use CPU."
    )


# ============================================================
# MODEL
# ============================================================

print(
    "\nCreating GRU model..."
)

model = GRURanker(
    vocab_size=len(
        vocab.itos
    ),
    emb_dim=128,
    hidden_dim=160
)

model = model.to(
    device
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# LOSS
# ============================================================

criterion = torch.nn.BCEWithLogitsLoss()


# ============================================================
# TRAINING
# ============================================================

print(
    "\nStarting training..."
)

print(
    "-" * 65
)

best_loss = float(
    "inf"
)

best_state = None


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0

    batch_count = 0

    for (
        input_ids,
        lengths,
        labels
    ) in loader:

        input_ids = input_ids.to(
            device
        )

        lengths = lengths.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            input_ids,
            lengths
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

        batch_count += 1

    average_loss = (
        total_loss /
        max(1, batch_count)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save best epoch
    # --------------------------------------------------------

    if average_loss < best_loss:

        best_loss = average_loss

        best_state = {
            key: value.cpu().clone()
            for key, value
            in model.state_dict().items()
        }


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaving improved RNN model..."
)

torch.save(
    {
        "state_dict":
            model.state_dict(),

        "stoi":
            vocab.stoi,

        "itos":
            vocab.itos
    },

    MODEL_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "-" * 65
)

print(
    "RNN / GRU TRAINING COMPLETED!"
)

print(
    f"Best training loss: "
    f"{best_loss:.4f}"
)

print(
    f"Model saved to:\n"
    f"{MODEL_FILE}"
)

print(
    "-" * 65
)