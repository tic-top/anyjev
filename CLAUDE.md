# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rule: every inference, sampling or experiment run logs as it goes

Any run that touches a model (server, parity, bench, eval, sampling, a one-off experiment script) must emit logs
**while it runs**: startup config, then a line per request, question, batch or step, then the result. A bug must
show up in the first few lines. Collecting everything and printing it only at the very end is **not allowed**.

- `print(..., flush=True)` or `python -u`. Redirected stdout is block-buffered, so an unflushed print reaches the log
  file only when the process exits. `scripts/parity.py` sets `sys.stdout.reconfigure(line_buffering=True)`.
- Long runs go to a log file (`> run.log 2>&1`, as in `scripts/bench.sh`) and get checked after the first few
  lines appear, not after the run finishes.
- Log the numbers you will judge the run by (per-question probs/argmax, ms, errors). "done" alone doesn't count.

## Rule: finished worktree → merge into main, push, delete it, no PRs

Small project, no review flow. When work in a worktree is done and checked, do all of this without asking:
commit in the worktree → leave it (ExitWorktree `keep`) → in the main checkout `git merge <branch>` →
`git push origin main` → `git worktree remove <path>` and `git branch -d <branch>`. Don't push the worktree branch.
**Never open a PR.** If the merge conflicts with uncommitted changes in the main checkout, stop and say so.

## Commands

```bash
pip install -e ".[test]"             # add hf,vision / mlx extras for the in-process backends
pytest tests                         # fast: fake backend + real tokenizers (what CI runs)
LLM2JEV_SLOW=1 pytest tests          # + real models through the transformers backend (GPU)
pytest tests/test_llm2jev.py::test_answers_and_warmup   # single test

llm2jev --model Qwen/Qwen3-0.6B --backend hf                                   # in-process reference server
llm2jev --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000  # against a running engine
python scripts/parity.py --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000
bash scripts/bench.sh <hf-model> <gpu> <name>   # JevBench + Decision Index via baby-jev ($BABY_JEV)
```

No linter or formatter is configured. Release: bump `version` in `pyproject.toml`, merge, publish a GitHub Release
(`.github/workflows/publish.yml`, PyPI trusted publishing).

## Architecture

llm2jev turns any chat model into a Jev `/v1/systemone` service: one prefill per question, and the answer is a softmax
over the logprobs of the option-label tokens at the single position after `Answer:`. It never samples.

Request flow, one direction:

- `__main__.py`: stdlib `ThreadingHTTPServer`. Maps `ValueError`/`KeyError`/`TypeError` to 422, a backend HTTP 400 to
  422 (e.g. over the context window), and any other backend failure to 504. `serve()` is reused by outside projects.
- `engine.py` `LLM2Jev`: at `__init__`, renders a probe prompt and runs `find_labels` so each label is checked to be
  exactly one new token *after the real prompt ending* (`A` vs ` A` comes from the tokenizer, not a guess). Per
  request: `render` once → `backend.warm(prefix)` → `score_many` (batched text, SGLang/vLLM) or `score` per question
  in a thread pool (media, in-process). A single question is scored on the caller's thread, without warming.
- `prompt.py`: the only place the wire format meets the model. The chat is rendered **once** with a random marker in
  the question slot, then split, so every question's prompt starts with byte-identical bytes and the engine prefix
  cache (SGLang radix / vLLM APC) serves them. Two styles: `chat` (model template, `enable_thinking=False`, assistant
  prefilled with `Answer:`; strict-alternation templates like Gemma get the question folded into the last user turn)
  and `jevlm` (raw completion prompt, text only).
- `backends.py`: `SGLang`, `VLLM`, `HF`, `MLX`, registered in `BACKENDS`. Contract: `warm(prefix, media)` and
  `score(text, media, ids) -> [logprob per id]`, optionally `score_many(texts, ids)`. A new engine is one class.
  Engines stay stock: SGLang `/generate` + `token_ids_logprob`; vLLM `/v1/completions` + `logprob_token_ids`
  (chunked, needs `--max-logprobs 256 --return-tokens-as-token-ids`), media via tokens-in `/inference/v1/generate`
  (`--enable-scale-out`). Missing/non-finite logprobs become `-inf`.
- `scoring.py`: `noul` → P(label `true`); `choice` → argmax, distribution, `confidence = 1 − H/log K`; `score` →
  expected level `Σ i·pᵢ`. `--temperature` rescales but never moves the argmax.

Limits: 1..64 questions per request, up to 255 labels (`A`–`Z`, then `AA`, `AB`, ...).

## Correctness = parity with the transformers reference

The `HF` backend is the reference. Any engine or model change is validated with `scripts/parity.py`: same rendered
prompts on both sides, every argmax must agree and every probability must be within `--tol` (0.03 default; 0.06 for
bf16 MoE). The README Status table records which engine version × model × modality has passed, plus known engine
bugs (SGLang 0.5.9 mis-scores Qwen3.6; SGLang audio needs 30 s clips; vLLM 0.26 ignores media). Update that table
when you add parity results.

Keep `render` and the label ids identical between serving and any training/RL code, so both score the same logit.
