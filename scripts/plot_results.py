from pathlib import Path
import csv
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = ROOT / "data" / "evaluation_results.csv"
OUTPUT_DIR = ROOT / "data" / "evaluation_graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL LABELS
# ============================================================

MODEL_LABELS = {
    "bm25": "BM25",
    "minilm": "All-MiniLM-L6-v2",
    "rnn": "RNN/GRU",
    "hybrid": "Hybrid",
}


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Evaluation results file not found:\n{RESULTS_FILE}"
    )


results = []

with open(RESULTS_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        results.append({
            "method": row["method"],
            "label": MODEL_LABELS.get(
                row["method"],
                row["method"]
            ),
            "exact_match": float(row["exact_match"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "latency": float(row["avg_latency_seconds"]),
        })


# ============================================================
# CHECK RESULTS
# ============================================================

if not results:
    raise ValueError("evaluation_results.csv is empty.")


print("\nLoaded evaluation results:")
print("-" * 70)

for row in results:
    print(
        f"{row['label']:<22}"
        f"EM={row['exact_match']:.4f}  "
        f"Precision={row['precision']:.4f}  "
        f"Recall={row['recall']:.4f}  "
        f"F1={row['f1']:.4f}  "
        f"Latency={row['latency']:.4f}s"
    )

print("-" * 70)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def create_graph(
    metric_key,
    title,
    y_label,
    filename,
    percentage=False
):
    labels = [row["label"] for row in results]
    values = [row[metric_key] for row in results]

    # Convert fractions to percentages
    if percentage:
        display_values = [value * 100 for value in values]
    else:
        display_values = values

    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, display_values)

    plt.title(title, fontsize=16)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.xticks(rotation=15)

    # Light horizontal grid
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # Add values above bars
    for bar, value in zip(bars, display_values):

        if percentage:
            text = f"{value:.2f}%"
        else:
            text = f"{value:.4f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# 1. EXACT MATCH
# ============================================================

create_graph(
    metric_key="exact_match",
    title="Exact Match Comparison",
    y_label="Exact Match (%)",
    filename="01_exact_match.png",
    percentage=True
)


# ============================================================
# 2. PRECISION
# ============================================================

create_graph(
    metric_key="precision",
    title="Precision Comparison",
    y_label="Precision (%)",
    filename="02_precision.png",
    percentage=True
)


# ============================================================
# 3. RECALL
# ============================================================

create_graph(
    metric_key="recall",
    title="Recall Comparison",
    y_label="Recall (%)",
    filename="03_recall.png",
    percentage=True
)


# ============================================================
# 4. F1 SCORE
# ============================================================

create_graph(
    metric_key="f1",
    title="F1 Score Comparison",
    y_label="F1 Score (%)",
    filename="04_f1_score.png",
    percentage=True
)


# ============================================================
# 5. AVERAGE RESPONSE LATENCY
# ============================================================

create_graph(
    metric_key="latency",
    title="Average Response Latency Comparison",
    y_label="Average Response Latency (seconds)",
    filename="05_average_response_latency.png",
    percentage=False
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ALL 5 EVALUATION GRAPHS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nOutput folder:")
print(OUTPUT_DIR)
print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {file.name}")

print("\nDone.")