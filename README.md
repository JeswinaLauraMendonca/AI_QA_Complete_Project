# AI-Based Question Answering System

A complete BE project implementation using:
- BM25: lexical retrieval
- All-MiniLM-L6-v2: semantic retrieval
- GRU/RNN: learned relevance ranking
- Hybrid: weighted combination of the three

Dataset: SQuAD v1.1, downloaded automatically from Hugging Face.

No paid API is required.

## Recommended setup

Python 3.10 or 3.11 is recommended.

### Windows PowerShell
```powershell
cd AI_QA_Complete
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_dataset.py
python scripts\build_index.py
python scripts\train_rnn.py
python backend\app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then open:
http://127.0.0.1:5000

## Linux/macOS
```bash
cd AI_QA_Complete
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/build_index.py
python scripts/train_rnn.py
python backend/app.py
```

## Evaluation

After the server is stopped, run:
```bash
python scripts/evaluate.py
```

The evaluation uses the SQuAD validation set and reports Exact Match and average latency for the four approaches.

## Project flow

Question -> BM25 / MiniLM / RNN -> Hybrid ranking -> best context -> extractive answer -> frontend

The first run downloads model weights. This is normal. GPU is optional; the code falls back to CPU.

## Complete evaluation

The project evaluates Exact Match, Precision, Recall, F1-score, Average Response Latency, and User Satisfaction (1-10 human survey). See `EVALUATION.md`.
