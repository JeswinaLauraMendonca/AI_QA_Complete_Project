from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

train_file = DATA / "squad_train.jsonl"
valid_file = DATA / "squad_validation.jsonl"

if train_file.exists() and valid_file.exists():
    print("SQuAD files already exist. Nothing to download.")
    raise SystemExit(0)

print("Downloading SQuAD v1.1...")
dataset = load_dataset("rajpurkar/squad")
dataset["train"].to_json(str(train_file))
dataset["validation"].to_json(str(valid_file))
print("Dataset saved to:", DATA)
