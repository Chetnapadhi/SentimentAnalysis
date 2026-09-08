"""Google Colab launcher for the E4 pipeline.

E4 = E0 text + PRETRAINED emoji embedding (dim 32, frozen, learned from TweetEval)
     with attention-based fusion → MLP head: Linear(800→256)→ReLU→Dropout(0.3)→Linear(256→3).

GUARDED: requires RUN_E4=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E4=1 to prevent accidental execution
if os.environ.get("RUN_E4") != "1":
    print("GUARDED: E4 requires RUN_E4=1 environment variable.")
    print("Run as: RUN_E4=1 python run_e4.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    from src.train_e4 import main as run_e4
    print(">>> Running E4 training + evaluation...")
    run_e4()


if __name__ == "__main__":
    main()
