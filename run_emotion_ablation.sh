#!/usr/bin/env bash
# Capacity-matched emoji ablation.
#
# Each run here is architecturally IDENTICAL to its counterpart in
# run_emotion_suite.sh -- same model, same parameter count, same cached text
# embeddings, same seed -- but every emoji input is zeroed, so each example
# takes the no-emoji path.
#
# The (real - ablated) gap is therefore attributable to emoji content alone.
# Comparing EM3 against EM0 instead would confound the emoji branch with a
# different classifier-head size (EM0 is 768->768, EM3/EM5 are 800->256).
#
# Cheap: frozen mode reuses the cached text embeddings, so each run is seconds.

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
    grep -E "TEST RESULTS|^Accuracy|^Macro F1|^Weighted F1|Emoji subset|ABLATION" "$log" | tail -8
    echo "--- ok: $log"
  else
    echo "!!! FAILED: see $log"
    tail -25 "$log"
  fi
}

for ds in tweeteval goemotions; do
  for model in em3 em5; do
    run "${ds}_${model}_frozen_ablate" \
      --dataset "$ds" --model "$model" --mode frozen --epochs 10 \
      --ablate-emoji --tag ablate
  done
done

echo ""
echo "##### ABLATION COMPLETE ($(date +%H:%M:%S)) #####"
