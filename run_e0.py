"""Google Colab launcher for the full E0 pipeline.

This orchestrates: dataset ingestion -> canonical build -> featurization ->
head training -> test evaluation -> plots -> error analysis -> research report.
Run the cells in ``colab/colab_e0_notebook.ipynb`` or execute this script after
the dataset build steps have run.

Requires a GPU runtime (T4) for reasonable featurization speed on 91K examples.
GUARDED: requires RUN_E0=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E0=1 to prevent accidental execution
if os.environ.get("RUN_E0") != "1":
    print("GUARDED: E0 requires RUN_E0=1 environment variable.")
    print("Run as: RUN_E0=1 python run_e0.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    # 1. Build final canonical datasets (from approved final CSVs)
    from src.data.preprocessing import build_final_canonical
    print(">>> Building canonical final datasets...")
    build_final_canonical("data/processed", "data/processed/canonical")

    # 2. Run the E0 training + evaluation pipeline
    print(">>> Running E0 training + evaluation...")
    from src.train import main as run_e0
    run_e0()

    # 3. Generate research report
    from src.report import generate_e0_report
    print(">>> Generating E0 research report...")
    generate_e0_report()

    print("\n>>> E0 complete. Artifacts in results/E0/")


if __name__ == "__main__":
    main()
