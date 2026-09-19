from pathlib import Path
import csv
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]; INPUT=ROOT/'data'/'user_satisfaction.csv'
if not INPUT.exists():
    print('Create data/user_satisfaction.csv using data/user_satisfaction_template.csv, then run this script again.'); raise SystemExit(1)
scores=defaultdict(list)
with open(INPUT,newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row.get('method','').strip() and row.get('rating_1_to_10','').strip(): scores[row['method'].strip().lower()].append(float(row['rating_1_to_10']))
print('\nUser Satisfaction (1-10)\n'+'-'*35)
for method in ['bm25','minilm','rnn','hybrid']:
    v=scores.get(method,[]); print(f"{method:<10} {sum(v)/len(v):.2f}/10 (n={len(v)})" if v else f'{method:<10} No ratings')
