"""Colab / local launcher for the Phase 2 emotion experiments.

Follows the same guard convention as ``run_e0.py`` ... ``run_e5.py`` so an
experiment can never start by accident.

    RUN_EMOTION=1 python run_emotion.py --dataset goemotions --model em0 --mode finetune

Every flag is forwarded verbatim to ``src.train_emotion``; see that module for
the full list.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("RUN_EMOTION") != "1":
    print("GUARDED: emotion experiments require RUN_EMOTION=1.")
    print("Run as: RUN_EMOTION=1 python run_emotion.py --dataset goemotions --model em0")
    sys.exit(0)

# Project root (this file lives at the repository root).
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from src.train_emotion import main as run

    print(">>> Running Phase 2 emotion training + evaluation...")
    run()


if __name__ == "__main__":
    main()
