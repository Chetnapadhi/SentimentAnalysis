"""Google Colab launcher for the E5 pipeline.

E5 = E0 text + PRETRAINED emoji embedding (dim 32, frozen, learned from TweetEval)
     with gated fusion → MLP head: Linear(800→256)→ReLU→Dropout(0.3)→Linear(256→3).
     Includes gate analysis artifacts.

GUARDED: requires RUN_E5=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E5=1 to prevent accidental execution
if os.environ.get("RUN_E5") != "1":
    print("GUARDED: E5 requires RUN_E5=1 environment variable.")
    print("Run as: RUN_E5=1 python run_e5.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    from src.train_e5 import main as run_e5
    print(">>> Running E5 training + evaluation...")
    run_e5()


if __name__ == "__main__":
    main()
