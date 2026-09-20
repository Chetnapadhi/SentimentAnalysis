"""Aggregate Phase 2 emotion results into one comparison table.

Walks ``results/emotion/<dataset>/<EXPERIMENT>/metrics.json`` and emits both a
machine-readable summary and a Markdown table ready to paste into the README.

Usage:
    python -m src.analysis.emotion_summary
    python -m src.analysis.emotion_summary --markdown
"""

from __future__ import annotations

import argparse
import glob
import json
import os

RESULTS_ROOT = "results/emotion"

MODEL_BLURB = {
    "EM0": "text only (no emoji branch)",
    "EM3": "attention fusion (text-conditioned over emojis)",
    "EM5": "gated fusion (32-d gate)",
}


def collect(root: str = RESULTS_ROOT) -> list[dict]:
    """Load every metrics.json under the emotion results tree."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "metrics.json"))):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        parts = path.replace("\\", "/").split("/")
        dataset, experiment = parts[-3], parts[-2]
        cfg = m.get("config", {})
        rows.append({
            "dataset": dataset,
            "experiment": experiment,
            "model": experiment.split("_")[0],
            "mode": cfg.get("mode", experiment.split("_", 1)[-1]),
            "backbone": cfg.get("backbone", "?"),
            "unfreeze_layers": cfg.get("unfreeze_layers", 0),
            "num_labels": len(m.get("label_names", [])),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "best_epoch": m.get("best_epoch"),
            "best_val_macro_f1": m.get("best_val_macro_f1"),
            "train_time_sec": m.get("training_time_sec"),
            "majority_acc": m.get("majority_baseline", {}).get("accuracy"),
            "majority_macro_f1": m.get("majority_baseline", {}).get("macro_f1"),
            "acc_with_emoji": m.get("accuracy_with_emoji"),
            "acc_without_emoji": m.get("accuracy_without_emoji"),
            "n_with_emoji": m.get("n_with_emoji"),
            "n_without_emoji": m.get("n_without_emoji"),
            "trainable_pct": m.get("trainable_params", {}).get("trainable_pct"),
            "path": os.path.dirname(path),
        })
    return rows


def _fmt(v, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def markdown_tables(rows: list[dict]) -> str:
    """Per-dataset Markdown tables, frozen runs first then fine-tuned."""
    out: list[str] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        drows = [r for r in rows if r["dataset"] == dataset]
        drows.sort(key=lambda r: (r["mode"] != "frozen", r["model"]))
        n_labels = drows[0]["num_labels"] if drows else 0
        maj = drows[0]["majority_acc"] if drows else None
        maj_f1 = drows[0]["majority_macro_f1"] if drows else None

        out.append(f"#### {dataset} ({n_labels}-class)")
        out.append("")
        out.append(
            "| Experiment | Mode | Emoji branch | Accuracy | Macro-F1 | "
            "Weighted-F1 | Best epoch | Train (s) |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        out.append(
            f"| _majority baseline_ | — | — | {_fmt(maj)} | {_fmt(maj_f1)} | — | — | — |"
        )
        for r in drows:
            branch = "no" if r["model"] == "EM0" else "yes"
            mode = "frozen" if r["mode"] == "frozen" else f"finetune top-{r['unfreeze_layers']}"
            out.append(
                f"| {r['experiment']} | {mode} | {branch} | "
                f"{_fmt(r['accuracy'])} | {_fmt(r['macro_f1'])} | "
                f"{_fmt(r['weighted_f1'])} | {r['best_epoch']} | "
                f"{r['train_time_sec']} |"
            )
        out.append("")

        out.append("Emoji-stratified test accuracy (same runs, test split split by emoji presence):")
        out.append("")
        out.append("| Experiment | With emoji | Without emoji | n(with) | n(without) |")
        out.append("|---|---|---|---|---|")
        for r in drows:
            out.append(
                f"| {r['experiment']} | {_fmt(r['acc_with_emoji'])} | "
                f"{_fmt(r['acc_without_emoji'])} | {r['n_with_emoji']} | "
                f"{r['n_without_emoji']} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=RESULTS_ROOT)
    p.add_argument("--markdown", action="store_true", help="Print Markdown tables only.")
    p.add_argument("--out", default=os.path.join(RESULTS_ROOT, "summary.json"))
    args = p.parse_args()

    rows = collect(args.root)
    if not rows:
        print(f"No metrics.json found under {args.root}/. Run the suite first.")
        return

    if args.markdown:
        print(markdown_tables(rows))
        return

    print(f"{'dataset':<12} {'experiment':<20} {'acc':>7} {'macroF1':>8} "
          f"{'maj.acc':>8} {'+emoji':>8} {'-emoji':>8}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: (r["dataset"], r["mode"] != "frozen", r["model"])):
        print(
            f"{r['dataset']:<12} {r['experiment']:<20} "
            f"{_fmt(r['accuracy'], 4):>7} {_fmt(r['macro_f1'], 4):>8} "
            f"{_fmt(r['majority_acc'], 4):>8} "
            f"{_fmt(r['acc_with_emoji'], 4):>8} {_fmt(r['acc_without_emoji'], 4):>8}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {args.out}")
    print("\nMarkdown tables:\n")
    print(markdown_tables(rows))


if __name__ == "__main__":
    main()
