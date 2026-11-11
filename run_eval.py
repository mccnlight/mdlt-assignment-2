"""Run both models over a ticket CSV and append raw results to JSONL (resumable).

usage: python run_eval.py [s1|llm|both] [data/tickets.csv]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

import llm_client
import workflow

RUNNERS = {"s1": workflow.decide_system_one, "llm": workflow.decide_llm}


def run(model, csv_path):
    tickets = pd.read_csv(csv_path)
    out_path = Path("results") / f"{Path(csv_path).stem}_{model}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines()}

    for row in tickets.itertuples():
        if row.id in done:
            continue
        for attempt in range(5):
            try:
                result = RUNNERS[model](row.text)
                break
            except llm_client.DailyLimitError as e:
                sys.exit(f"{model} {row.id}: daily quota exhausted, stopping. {e}")
            except Exception as e:  # rate limits / transient provider errors
                print(f"{model} {row.id} attempt {attempt + 1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"{model} {row.id}: giving up", flush=True)
            continue
        result["id"] = row.id
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"{model} {row.id}: {result.get('route', 'INVALID')} ({result['latency']:.1f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "data/tickets.csv"
    for m in (["s1", "llm"] if which == "both" else [which]):
        run(m, csv_path)
