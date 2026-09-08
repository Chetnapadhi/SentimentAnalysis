"""Google Colab launcher for the E3 pipeline.

E3 = E0 text + random emoji embedding (dim 32) with attention-based fusion,
then MLP head: Linear(800→256)→ReLU→Dropout(0.3)→Linear(256→3).

GUARDED: requires RUN_E3=1 environment variable to execute.
"""

from __future__ import annotations

import os
import sys

# GUARD: require RUN_E3=1 to prevent accidental execution
if os.environ.get("RUN_E3") != "1":
    print("GUARDED: E3 requires RUN_E3=1 environment variable.")
    print("Run as: RUN_E3=1 python run_e3.py")
    sys.exit(0)

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    from src.train_e3 import main as run_e3
    print(">>> Running E3 training + evaluation...")
    run_e3()


if __name__ == "__main__":
    main()
