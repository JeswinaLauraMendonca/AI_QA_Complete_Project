# Evaluation Plan

The system evaluates four approaches: BM25, All-MiniLM-L6-v2, RNN/GRU, and Hybrid.

Objective metrics:
1. Exact Match (EM)
2. Precision
3. Recall
4. F1-score
5. Average Response Latency

Run `python scripts/evaluate.py`. It evaluates 100 SQuAD validation questions by default and saves `data/evaluation_results.csv`.

User Satisfaction is a human evaluation on a 1-10 scale. Copy `data/user_satisfaction_template.csv` to `data/user_satisfaction.csv`, collect real ratings for every approach, then run `python scripts/user_satisfaction.py`.

Do not invent or copy satisfaction scores. The first research paper uses accuracy, response time, and user satisfaction among its evaluation measures; its reported participant count is not automatically the participant count for this project.
