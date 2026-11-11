"""Score results against gold labels: results table + confidence-threshold plot."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import workflow

# Clef list price, USD per 1M input tokens (output is free). Our runs used the free tier.
CLEF_PRICE_PER_M_INPUT = 0.24
NAMES = {"s1": "Clef 27B (System One)", "llm": "gpt-oss-120b (LLM, JSON)"}
TARGET = 0.95


def load(model, stem="tickets"):
    gold = pd.read_csv(f"data/{stem}.csv")
    gold["gold_route"] = [workflow.route(d, p) for d, p in zip(gold.department, gold.priority)]
    path = Path("results") / f"{stem}_{model}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    pred = pd.DataFrame(rows)
    df = gold.merge(pred, on="id", how="left", suffixes=("_gold", ""))
    df["valid"] = df["valid"].fillna(False).astype(bool)  # missing call = invalid
    df["route_ok"] = df.valid & (df.route == df.gold_route)
    df["dept_ok"] = df.valid & (df.department == df.department_gold)
    df["prio_ok"] = df.valid & (df.priority == df.priority_gold)
    df["esc_ok"] = df.valid & (df.escalate == df.escalate_gold)
    return df


def cost_per_1000(df, model):
    if model == "s1":
        return (df["input_tokens"] * CLEF_PRICE_PER_M_INPUT / 1e6).mean() * 1000
    return df["cost"].mean() * 1000  # measured from Workers AI neurons at list price


def results_table():
    rows = []
    for m in ["s1", "llm"]:
        df = load(m)
        rows.append({
            "model": NAMES[m],
            "route acc": df.route_ok.mean(),
            "department acc": df.dept_ok.mean(),
            "priority acc": df.prio_ok.mean(),
            "escalate acc": df.esc_ok.mean(),
            "avg latency (s)": df.latency.mean(),
            "p95 latency (s)": df.latency.quantile(0.95),
            "cost / 1k decisions ($)": cost_per_1000(df, m),
            "invalid %": 100 * (1 - df.valid.mean()),
        })
    return pd.DataFrame(rows).set_index("model")


def confidence_curve():
    df = load("s1").sort_values("confidence", ascending=False)
    thresholds = np.round(np.arange(0.0, 1.0, 0.01), 2)
    acc, cov = [], []
    for t in thresholds:
        sel = df[df.confidence >= t]
        acc.append(sel.route_ok.mean() if len(sel) else np.nan)
        cov.append(len(sel) / len(df))
    curve = pd.DataFrame({"threshold": thresholds, "accuracy": acc, "coverage": cov})
    # Lowest threshold from which accuracy stays >= target for every higher threshold.
    below = curve.index[curve.accuracy.fillna(1.0) < TARGET]
    start = below.max() + 1 if len(below) else 0
    best = curve.loc[start] if start < len(curve) else None

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve.threshold, curve.accuracy, label="accuracy of auto-decided tickets", color="#2b6cb0", lw=2)
    ax.plot(curve.threshold, curve.coverage, label="share auto-decided (coverage)", color="#dd6b20", lw=2)
    ax.axhline(TARGET, color="gray", ls="--", lw=1)
    ax.text(0.01, TARGET + 0.01, "95% accuracy target", color="gray", fontsize=9)
    if best is not None:
        ax.axvline(best.threshold, color="#2f855a", ls=":", lw=1.5)
        ax.annotate(f"threshold {best.threshold:.2f}\n{best.coverage:.0%} auto, {1 - best.coverage:.0%} to human",
                    (best.threshold, best.coverage), xytext=(10, -40), textcoords="offset points", fontsize=9)
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Clef: accuracy vs confidence threshold (route, 80 tickets)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/confidence_threshold.png", dpi=150)
    return curve, best


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    table = results_table()
    print(table.round(3).to_string())
    curve, best = confidence_curve()
    if best is not None:
        print(f"\nlowest threshold with >= {TARGET:.0%} accuracy: {best.threshold:.2f} "
              f"-> {best.coverage:.0%} auto-decided, {1 - best.coverage:.0%} to a human")
    else:
        print(f"\nno threshold reaches {TARGET:.0%} accuracy")
    s1 = load("s1")
    print("\nClef errors:")
    print(s1[~s1.route_ok][["id", "gold_route", "route", "confidence"]].to_string(index=False))
