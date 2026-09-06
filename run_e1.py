"""Google Colab launcher for the E1 pipeline.

E1 = E0 text + random emoji embedding (dim 32) mean-pooled, concatenated,
then MLP head: Linear(800->256)->ReLU->Dropout(0.3)->Linear(256->3).

GUARDED: requires RUN_E1=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E1=1 to prevent accidental execution
if os.environ.get("RUN_E1") != "1":
    print("GUARDED: E1 requires RUN_E1=1 environment variable.")
    print("Run as: RUN_E1=1 python run_e1.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    # Build emoji vocab from train, featurize text, train head+emoji, evaluate
    from src.train_e1 import main as run_e1
    print(">>> Running E1 training + evaluation...")
    run_e1()


if __name__ == "__main__":
    main()