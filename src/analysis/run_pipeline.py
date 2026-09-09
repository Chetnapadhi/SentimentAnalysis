"""Master execution script for the final analysis pipeline.

Generates all tables, figures, and artifacts under results/final_analysis/.
Does NOT retrain models, alter existing metrics, or touch original E0–E5 folders.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure DL root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.data_loader import (
    get_project_root,
    load_all_confusion_matrices,
    load_all_predictions,
)
from src.analysis.emoji_stratification import (
    analyze_by_emoji_count,
    analyze_emoji_presence,
    analyze_sentiment_by_emoji_count,
    plot_emoji_presence_comparison,
)
from src.analysis.error_transitions import analyze_error_transitions
from src.analysis.gate_interpretation import (
    build_gate_summary_table,
    load_gate_data,
    plot_gate_analysis,
)
from src.analysis.master_metrics import (
    build_master_comparison,
    build_model_rankings,
    plot_master_performance,
)
from src.analysis.per_class_and_cm import (
    extract_per_class_metrics,
    plot_all_confusion_matrices,
    plot_per_class_metrics,
)


def run_pipeline() -> Path:
    """Run all analysis steps and save outputs to results/final_analysis/."""
    root = get_project_root()
    out_dir = root / "results" / "final_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"RUNNING FINAL REPRODUCIBLE ANALYSIS PIPELINE -> {out_dir}")
    print("=" * 70)

    # 1. Master comparison & rankings
    print("\n[Step 1/6] Building Master Comparison & Model Rankings...")
    master_df = build_master_comparison()
    rankings_df = build_model_rankings(master_df)

    master_df.to_csv(out_dir / "master_results.csv", index=False)
    rankings_df.to_csv(out_dir / "model_rankings.csv", index=False)

    master_dict = master_df.to_dict(orient="records")
    with open(out_dir / "master_results.json", "w", encoding="utf-8") as f:
        json.dump(master_dict, f, indent=2)

    plot_master_performance(rankings_df, out_dir / "performance_comparison.png")
    print("  [OK] Saved master_results.csv, master_results.json, model_rankings.csv")
    print("  [OK] Saved performance_comparison.png")

    # 2. Per-class metrics & confusion matrices
    print("\n[Step 2/6] Extracting Per-Class Metrics & Plotting Confusion Matrices...")
    per_class_df = extract_per_class_metrics()
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    plot_per_class_metrics(per_class_df, out_dir / "per_class_comparison.png")

    cms = load_all_confusion_matrices()
    plot_all_confusion_matrices(cms, out_dir / "confusion_matrices_all.png")
    print("  [OK] Saved per_class_metrics.csv")
    print("  [OK] Saved per_class_comparison.png, confusion_matrices_all.png")

    # 3. Load merged predictions for sample-level analyses
    print("\n[Step 3/6] Loading Merged Predictions (N = 11,966)...")
    preds_df = load_all_predictions()
    print(f"  [OK] Successfully merged canonical test metadata with E0–E5 predictions (Shape: {preds_df.shape})")

    # 4. Emoji stratification (presence & frequency)
    print("\n[Step 4/6] Analyzing Performance Stratified by Emojis...")
    presence_df = analyze_emoji_presence(preds_df)
    presence_df.to_csv(out_dir / "emoji_presence_analysis.csv", index=False)

    count_df = analyze_by_emoji_count(preds_df)
    count_df.to_csv(out_dir / "emoji_count_breakdown.csv", index=False)

    sentiment_dist_df = analyze_sentiment_by_emoji_count(preds_df)
    sentiment_dist_df.to_csv(out_dir / "sentiment_by_emoji_count.csv", index=False)

    plot_emoji_presence_comparison(presence_df, out_dir / "emoji_performance_comparison.png")
    print("  [OK] Saved emoji_presence_analysis.csv, emoji_count_breakdown.csv, sentiment_by_emoji_count.csv")
    print("  [OK] Saved emoji_performance_comparison.png")

    # 5. Error transition analysis
    print("\n[Step 5/6] Performing Error Transition Analysis (E0 vs E3 vs E5)...")
    trans_summary, trans_examples = analyze_error_transitions(preds_df)
    trans_summary.to_csv(out_dir / "transition_summary.csv", index=False)
    trans_examples.to_csv(out_dir / "error_transition_cases.csv", index=False)
    print("  [OK] Saved transition_summary.csv, error_transition_cases.csv")

    # 6. E5 Gate analysis
    print("\n[Step 6/6] Analyzing E5 32-Dimensional Gate Values...")
    gate_data = load_gate_data()
    gate_summary = build_gate_summary_table(gate_data)
    gate_summary.to_csv(out_dir / "gate_summary_table.csv", index=False)
    plot_gate_analysis(gate_data, out_dir / "e5_gate_analysis.png")
    print("  [OK] Saved gate_summary_table.csv, e5_gate_analysis.png")

    print("\n" + "=" * 70)
    print("FINAL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All outputs consolidated in: {out_dir}")
    print("=" * 70)
    return out_dir


if __name__ == "__main__":
    run_pipeline()
