"""Google Colab launcher for the E2 pipeline.

E2 = E0 text + PRETRAINED emoji embedding (dim 32, frozen, learned from TweetEval)
     concatenated → MLP head: Linear(800→256)→ReLU→Dropout(0.3)→Linear(256→3).

GUARDED: requires RUN_E2=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E2=1 to prevent accidental execution
if os.environ.get("RUN_E2") != "1":
    print("GUARDED: E2 requires RUN_E2=1 environment variable.")
    print("Run as: RUN_E2=1 python run_e2.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    # Build emoji vocab from train, featurize text, train head, evaluate
    from src.train_e2 import main as run_e2
    print(">>> Running E2 training + evaluation...")
    run_e2()


if __name__ == "__main__":
    main()