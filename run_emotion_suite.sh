#!/usr/bin/env bash
# Phase 2 emotion tournament driver.
#
# Runs the full EM0 / EM3 / EM5 grid over both emotion corpora, cheapest first,
# so partial results are available early. Each run writes its own
# results/emotion/<dataset>/<EXPERIMENT>/metrics.json; this script only
# sequences them and tees per-run logs.
#
# Usage:  bash run_emotion_suite.sh [stage]
#   stage = tweeteval | goemotions | all   (default: all)

set -u

PY=".venv/Scripts/python.exe"
LOGDIR="results/emotion/_logs"
mkdir -p "$LOGDIR"

export PYTHONIOENCODING=utf-8
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

run() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo ""
  echo "=================================================================="
  echo ">>> $name   ($(date +%H:%M:%S))"
  echo "=================================================================="
  if $PY -m src.train_emotion "$@" >"$log" 2>&1; then
    grep -E "TEST RESULTS|^Accuracy|^Macro F1|^Weighted F1|Majority baseline|Emoji subset|Trainable:|Unfroze" "$log" | tail -12
    echo "--- ok: $log"
  else
    echo "!!! FAILED: see $log"
    tail -25 "$log"
  fi
}

STAGE="${1:-all}"

if [ "$STAGE" = "tweeteval" ] || [ "$STAGE" = "all" ]; then
  echo "##### STAGE 1: TweetEval emotion (4-class, emoji-rich) #####"
  run tweeteval_em0_frozen   --dataset tweeteval --model em0 --mode frozen   --epochs 10
  run tweeteval_em3_frozen   --dataset tweeteval --model em3 --mode frozen   --epochs 10
  run tweeteval_em5_frozen   --dataset tweeteval --model em5 --mode frozen   --epochs 10
  run tweeteval_em0_finetune --dataset tweeteval --model em0 --mode finetune --epochs 4 --unfreeze-layers 2
  run tweeteval_em3_finetune --dataset tweeteval --model em3 --mode finetune --epochs 4 --unfreeze-layers 2
fi

if [ "$STAGE" = "goemotions" ] || [ "$STAGE" = "all" ]; then
  echo ""
  echo "##### STAGE 2: GoEmotions Ekman-6 (6-class, the supervisor's target) #####"
  run goemotions_em0_frozen   --dataset goemotions --model em0 --mode frozen   --epochs 10
  run goemotions_em3_frozen   --dataset goemotions --model em3 --mode frozen   --epochs 10
  run goemotions_em5_frozen   --dataset goemotions --model em5 --mode frozen   --epochs 10
  run goemotions_em0_finetune --dataset goemotions --model em0 --mode finetune --epochs 3 --unfreeze-layers 2
  run goemotions_em3_finetune --dataset goemotions --model em3 --mode finetune --epochs 3 --unfreeze-layers 2
fi

echo ""
echo "##### SUITE COMPLETE ($(date +%H:%M:%S)) #####"
