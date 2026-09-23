# Architecture

llm2jev is ~500 lines in five modules. A request flows through them in one direction:

```
POST /v1/systemone                      __main__.py   threaded HTTP server (backlog 1024), 422 client / 504 backend
   │ state, questions
   ▼
LLM2Jev.run → (answers, usage)          engine.py     validate, render, warm prefix, ONE engine call per request
   │
   ├─▶ render(processor, state, qs)      prompt.py     ONE chat-template render with a marker in the question slot
   │      → prefix (shared bytes)                      → prefix + one suffix per question, media list
   │      → {qid: prompt, answer keys}
   │
   ├─▶ backend.warm(prefix, media)       backends.py   one request so the engine's prefix cache holds the state
   ├─▶ backend.score_many(prompts, ids)  backends.py   all questions in one /generate (SGLang) or /v1/completions (vLLM)
   │      (1 question: backend.score)                  call: one prefill each, max_new_tokens=1, label logprobs
   ▼
answer(question, keys, softmax(ℓ/T))    scoring.py    noul P(Yes) · choice argmax+dist+confidence · score E[level]
```

## Modules

| file | owns | key functions |
|---|---|---|
| `prompt.py` | the prompt, the only place the wire format meets the model | `render`, `state_messages`, `options_of`, `find_labels` |
| `engine.py` | one request end to end | `LLM2Jev.__init__` (label check), `LLM2Jev.run -> (answers, usage)`, `LLM2Jev.__call__ -> answers` |
| `backends.py` | talking to an engine | `SGLang`, `VLLM`, `HF`, `MLX`: `warm(prefix, media)`, `score(text, media, ids) -> (logprobs, prompt tokens)`; engine backends add `score_many(texts, ids) -> (rows, prompt tokens)` |
| `scoring.py` | logprobs → Jev answers | `softmax`, `confidence`, `answer` |
| `__main__.py` | CLI and HTTP | `main`, `Server`, `serve` (also used by projects that build their own `LLM2Jev`) |

Adding an engine means one class with `warm` and `score` (plus `score_many` if the engine takes a batch of prompts).
Nothing else changes.

## Design decisions

**All options in one prompt, one readout position.** Each question lists every option under single-token labels
(`A. …`, `B. …`), and the distribution is the softmax of those label logprobs at the token after `Answer:`. Options
compete inside the model, so "none of the above" or "the more specific one" can depend on the other options. That
matches what black-box probing found about Jev itself ([archerhume][archer]: options interact, order matters,
255-option cap). It costs one forward per question, whatever the option count.

**The model's own chat template, thinking off.** Instruct models are trained to answer a user turn, so the question
goes in a last user turn and the assistant turn is pre-filled with `Answer:`. Templates that insist on strict
user/assistant alternation (Gemma) get the question folded into the last user turn instead.

**Labels are checked in context, at startup.** Whether the next token is `A` or ` A` depends on the template ending.
`find_labels` tokenizes the real prompt ending plus each candidate label and keeps those that are exactly one new token.
Nothing is guessed per tokenizer family.

**The shared prefix is byte-identical.** The chat is rendered once with a random marker where the question goes, then
split. Every question's prompt starts with the same bytes, one warm request fills the prefix cache (SGLang radix, vLLM
APC), and the question requests only prefill their own suffix.

**One engine call per Jev request; the engine schedules.** A text request with several questions is sent as ONE
batched call, a single question runs on the request's own thread, and the thread pool only serves media requests and
in-process backends. Measured on Qwen3-1.7B / SGLang 0.5.9 / A100, same load for every variant (1500 Decision Index
rows at 64 concurrent clients; 300 rows of 17+ questions at 16):

| client | rows/s (mixed) | questions/s (17+ q) | errors |
|---|---|---|---|
| one request per question, 16-thread pool, backlog 5 (≤ 0.5.0) | 21.1 | 124.7 | connection resets under bursts |
| **one call per Jev request (this design)** | **24.6** | **148.7** | 0 |
| async fan-out, one request per question, 64 in flight | 14.9 | 85.1 | 0 |
| async fan-out, 256 in flight | 8.5 | 43.7 | 504s |
| SGLang `/v1/score` (engine-side fan-out) | 22.7 | 145.5 | 0 |
| SGLang `/v1/score` multi-item (one masked forward) | 30.3 | 206.8 | 422 on long requests; answers change |

Flooding the engine with one request per question is the slow path: its scheduler is single-threaded and pays per
queued request. Multi-item packing is the fastest, but it puts a delimiter token the model can see in front of every
question, so answers move unless the model is trained on that format (and it cannot isolate items in hybrid
linear-attention models such as Qwen3.5).

**Engines are untouched.** Every backend uses a stock API (SGLang `/generate` with `token_ids_logprob`, vLLM
`/v1/completions` with `logprob_token_ids`, or the tokens-in `/inference/v1/generate` for media). Parity against the
in-process transformers reference is checked label by label (`scripts/parity.py`).

**One knob: temperature.** `--temperature` rescales the distribution and never moves the argmax. There is no
per-benchmark prompt, few-shot example or option filtering.

## How this differs from other Jev reproductions

Most open Jev reproductions train something. llm2jev asks how far a stock chat model already gets when it is only
*read* correctly. Grouped by what they change:

| project | trains? | options seen together? | forwards per question | readout | engines |
|---|---|---|---|---|---|
| **llm2jev** (this repo) | no | yes, one prompt | 1 | label-token logprob after `Answer:` | SGLang, vLLM, transformers, MLX |
| [Yinsongxu/LLM2Jev][ysx] (same name, unrelated) | no | no, one yes/no prompt per option | 1 per option (prefix-cached) | softmax(yes, no), renormalized over options | SGLang, transformers |
| [ekzhang/openjev-sglang][oj] | no | yes, one prompt | 1 | label-token logprob | SGLang, one model (Qwen3.6-35B-A3B NVFP4) |
| [kikoncuo/jevfire][jf], [genai-craft/openvons][ov], [r-ms/mini-jev][mj] | no | — | — | inference techniques on Qwen3.8-27B / Qwen3-4B-Instruct-2507 | — |
| [tic-top/baby-jev][bj] | LoRA | yes | 1 | same label readout, fitted T; **served through llm2jev** | via llm2jev |
| [jaredpalmer/kev][kev] | LoRA + pointer head | yes (optional isolation) | 1 | custom head over option tokens | own server |
| [Zefan-Cai/Open-Jev][ojz], [IamBusy/OpenJev-Vision][ojv] | LoRA + scalar head | no, one branch per option | 1 per option | scalar per candidate | own server |
| [bespokelabsai/nimble][nim], Decider, NanoJev, … | LoRA / full fine-tune | — | — | — | — |

The per-option designs ([Yinsongxu/LLM2Jev][ysx], OpenJev-Vision, Open-Jev) are order-independent by construction but
cannot let one option change another's probability. The trained designs buy accuracy and calibration with data and a
custom serving stack. llm2jev keeps the model and the engine stock, which makes it a zero-shot baseline for any new
chat model on release day and a serving layer for fine-tunes that use the same readout (baby-jev does exactly that).

[archer]: https://archerhume.com/posts/jevs-architecture-unmasked
[ysx]: https://github.com/Yinsongxu/LLM2Jev
[oj]: https://github.com/ekzhang/openjev-sglang
[jf]: https://github.com/kikoncuo/jevfire
[ov]: https://github.com/genai-craft/openvons
[mj]: https://github.com/r-ms/mini-jev
[bj]: https://github.com/tic-top/baby-jev
[kev]: https://github.com/jaredpalmer/kev
[ojz]: https://github.com/Zefan-Cai/Open-Jev
[ojv]: https://github.com/IamBusy/OpenJev-Vision
[nim]: https://github.com/bespokelabsai/nimble
