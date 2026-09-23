#!/usr/bin/env bash
# Zero-shot JevBench + Decision Index for one HF chat model on one GPU, through llm2jev, scored by baby-jev's harness.
#   bash scripts/bench.sh Qwen/Qwen3.5-4B 0 qwen3.5-4b [--di-rows index-rows.jsonl.gz]
# SGLang on 31000+gpu, llm2jev on 18080+gpu; results in $BABY_JEV/results/bench/llm2jev-<name>/. The harness is
# jev.eval.bench from https://github.com/tic-top/baby-jev (fstandhartinger/jevbench + kshetrajna12/decision-index).
set -uo pipefail
model=$1 g=$2 name=$3; shift 3
BJ=${BABY_JEV:-$HOME/baby-jev}; here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
logs=${LOGS:-$BJ/logs/llm2jev-$name}; mkdir -p "$logs"
sp=$((31000 + g)) ap=$((18080 + g))
case $model in  # hybrid (Gated DeltaNet) models need SGLang's mamba prefix cache; plain transformers must not get it
  *Qwen3.5*) extra=(--mamba-scheduler-strategy extra_buffer) ;;
  *gemma*) extra=(--context-length 32768) ;;  # 128k KV + vision tower does not fit a 40 GB card at 0.8
  *) extra=() ;;
esac
stop() { for p in "$@"; do pkill -TERM -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; sleep 8
         fuser -k -9 $sp/tcp $ap/tcp >/dev/null 2>&1; }
CUDA_VISIBLE_DEVICES=$g bash "$BJ/scripts/with-jev-env.sh" sglang -m sglang.launch_server --model-path "$model" \
  --host 127.0.0.1 --port $sp --mem-fraction-static 0.8 "${extra[@]}" > "$logs/sglang.log" 2>&1 &
spid=$!
until curl -sf localhost:$sp/health >/dev/null; do kill -0 $spid 2>/dev/null || { echo "$name: sglang died"; exit 1; }; sleep 2; done
PYTHONPATH=$here bash "$BJ/scripts/with-jev-env.sh" hf -m llm2jev --model "$model" --url http://127.0.0.1:$sp --port $ap \
  > "$logs/api.log" 2>&1 &
apid=$!
until curl -sf localhost:$ap/health >/dev/null; do kill -0 $apid 2>/dev/null || { echo "$name: api died"; stop $spid; exit 1; }; sleep 2; done
(cd "$BJ" && bash scripts/with-jev-env.sh hf -m jev.eval.bench run --name "llm2jev-$name" --url http://127.0.0.1:$ap \
  --only jevbench,di --per-url 24 "$@" > "$logs/bench.log" 2>&1)
rc=$?
stop $apid $spid
echo "$name done rc=$rc"
exit $rc
