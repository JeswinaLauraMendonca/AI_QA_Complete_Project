from pathlib import Path
import sys

# Make the project root importable even when launched as: python backend/app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from backend.qa_engine import QAEngine

app = Flask(__name__)
CORS(app)

FRONTEND = ROOT / "frontend"
engine = None

@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/ask")
def ask():
    global engine

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    method = str(data.get("method", "hybrid")).lower()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    allowed = {"bm25", "minilm", "rnn", "hybrid"}
    if method not in allowed:
        return jsonify({"error": "Choose BM25, All-MiniLM, RNN or Hybrid."}), 400

    try:
        if engine is None:
            engine = QAEngine()
        results = engine.query(question, method, 5)
        if not results:
            return jsonify({"error": "No result found."}), 404
        return jsonify({
            "question": question,
            "method": method,
            "best": results[0],
            "results": results
        })
    except Exception as exc:
        app.logger.exception("QA request failed")
        return jsonify({
            "error": "The QA engine could not process the request.",
            "details": str(exc)
        }), 500

if __name__ == "__main__":
    print("QA system running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
