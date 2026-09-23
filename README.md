# llm2jev

[中文](https://github.com/tic-top/llm2jev/blob/main/README.zh-CN.md)

Turn **any chat model** into a [Jev](https://docs.typesafe.ai/api)-compatible probability decision service, on
**SGLang, vLLM, plain transformers or MLX (Apple silicon)**, with no model or engine changes.

```
state + question + ALL options  ──one prefill──▶  logprobs of the option labels at one position  ──▶  probabilities
```

- **One forward per question**, however many options (up to 255). No generation, no sampling.
- **All options in one prompt**, so options compete directly ("none of the above" can depend on the others).
- **Standard chat template, thinking off.** Works with instruct models zero-shot; fine-tuned checkpoints use the same prompt.
- **Shared prefix, rendered once.** Every question of a request starts with byte-identical text, the prefix is warmed
  once, and the engine's prefix cache (SGLang radix / vLLM APC) serves the rest.
- **Images, video and audio in the state** (SGLang, vLLM and transformers backends) for multimodal models.

llm2jev is an independent implementation of the documented System One wire format. It is not affiliated with TypeSafe.

It was called AnyJev before 0.5.0.

## Quick start

```bash
pip install llm2jev              # client + server; the engine runs separately
pip install "llm2jev[hf,vision]" # + in-process transformers backend (video and audio decoders included)
pip install "llm2jev[mlx]"       # + in-process MLX backend (Apple silicon)

# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3.5-2B --port 30000
llm2jev --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000

# vLLM (--enable-scale-out opens the tokens-in endpoint used for images, video and audio)
vllm serve Qwen/Qwen3.5-2B --max-logprobs 256 --return-tokens-as-token-ids --enable-scale-out --port 8000
llm2jev --model Qwen/Qwen3.5-2B --backend vllm --url http://127.0.0.1:8000

# transformers, in-process reference (slow, no prefix cache)
llm2jev --model Qwen/Qwen3-0.6B --backend hf

# MLX on Apple silicon, in-process (text only; HF ids or mlx-community quantized repos)
llm2jev --model mlx-community/Qwen3-0.6B-4bit --backend mlx
```

From Python, without the HTTP server:

```python
from transformers import AutoProcessor
from llm2jev import LLM2Jev
from llm2jev.backends import SGLang

jev = LLM2Jev(AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B"), SGLang("http://127.0.0.1:30000"))
jev(state, questions)  # {"refund": {"type": "noul", "noul": 0.97}, ...}
```

```bash
curl localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": [{"role": "system", "content": "You are a support assistant."},
            {"role": "user", "content": "I was charged twice. Please refund the duplicate."}],
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is the request?",
                   "criteria": ["Routine", "Urgent", "Emergency"]}}}'
```

Media: put `{"type": "image", "image": "<path | https URL | data: URI>"}` parts in a state message, likewise
`{"type": "video", "video": ...}` and `{"type": "audio", "audio": ...}` (or OpenAI-style `image_url` / `video_url` /
`audio_url`). The transformers backend reads video from a path or URL only (`llm2jev[hf]` brings the
video and audio decoders). Audio on vLLM needs `vllm[audio]`.

## How a score is computed

The request is rendered once with the model's own chat template (`enable_thinking=False`). The question goes in a last
user turn, and the assistant turn is pre-filled with `Answer:`:

```
<|im_start|>user
Evaluate the conversation or state above using the question below. ... reply with its label only.

Question: How urgent is the request?
Options:
A. 0: Routine
B. 1: Urgent
C. 2: Emergency<|im_end|>
<|im_start|>assistant
<think>

</think>

Answer:▸ read ' A' ' B' ' C'
```

The engine returns the full-vocabulary logprob ℓᵢ of each label token at that one position (`max_new_tokens=1`,
selected-token logprobs). Then p = softmax(ℓ / T) over this question's labels:

| type   | answer |
|--------|--------|
| noul   | `P(Yes)`. Options are `A. Yes`, `B. No`. |
| choice | argmax, the full distribution, and `confidence = 1 − H(p)/log K` |
| score  | expected level `Σ i·pᵢ` (levels from 0), the distribution, and confidence |

Labels are `A`–`Z`, then two-letter labels (`AA`, `AB`, ...). At startup, each label is **checked against the real
prompt ending** and must be exactly one new token there. Whether a model writes `'A'` or `' A'` after `Answer:` is
therefore decided by the tokenizer, not guessed.

`--temperature` rescales the distribution and never changes the argmax. Fit it once, globally, on held-out data if you
need calibrated probabilities.

## Status

Parity = `scripts/parity.py`: the engine and the transformers reference score the same rendered prompts. Every
question must agree on the argmax and stay within 0.03 in probability.

| backend | text | image | video | audio |
|---|---|---|---|---|
| transformers | reference (Qwen3-0.6B, Qwen3.5-2B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B, Qwen2-Audio-7B) | reference (Qwen3.5-2B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B) | reference (Qwen3.5-2B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B) ⁶ | reference (Qwen2-Audio-7B) |
| SGLang 0.5.18 | parity ✓ Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B ⁴ | parity ✓ Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B | parity ✓ Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B | – |
| SGLang 0.5.9 | parity ✓ Qwen3-0.6B, Qwen3.5-2B, Qwen2-Audio-7B | parity ✓ Qwen3.5-2B | parity ✓ Qwen3.5-2B ¹ | 30 s clips only ² |
| vLLM 0.30.0 | parity ✓ Qwen3-0.6B, Qwen3.5-2B, Qwen2-Audio-7B | parity ✓ Qwen3.5-2B | parity ✓ Qwen3.5-2B | parity ✓ Qwen2-Audio-7B |
| vLLM 0.26.0 | parity ✓ Qwen3.6-27B, Qwen3.6-35B-A3B ⁴ | ✗ ⁵ | ✗ ⁵ | – |
| MLX (mlx-lm) | matches transformers ✓ Qwen3-0.6B ³ | – | – | – |

¹ Within tolerance, but not byte-identical. For Qwen3.5 video, SGLang drops the template's outer
`<|vision_start|>…<|vision_end|>` around the per-frame blocks (2 tokens) that transformers and vLLM keep. It also
upscales small frames (128 px becomes about 320 px), so the parity clip is 320 px.
² SGLang 0.5.9 runs the Qwen2-Audio encoder without the feature attention mask, so a clip shorter than Whisper's 30 s
window attends to its zero padding. A 2 s clip is off by 0.20 in probability, and a 30 s clip matches. Use vLLM or
transformers for audio.
³ `test_mlx_matches_hf`, run on the mlx CPU build. No prefix cache: each question is a full prefill.
⁴ Qwen3.6 (27B dense, 35B-A3B MoE), TP 2 and TP 4. SGLang 0.5.9 loads Qwen3.6 but scores it wrong (off by up to 0.8
in probability on text), so use 0.5.18. It also mis-scores Qwen3.5-27B at TP 2 (JevBench public accuracy 0.39 vs
0.88 on 0.5.18). On 35B-A3B every argmax agrees, but one near-tie question (0.77 vs 0.22)
differs by 0.05 between any two of SGLang, vLLM and transformers. That is bf16 MoE routing noise, so run MoE parity
with `--tol 0.06`. Qwen3.8-27B passes with `--enable-fp32-lm-head` (bf16 head: 0.031 on one question).
⁵ vLLM 0.26 accepts `content_parts` on `/inference/v1/generate` but ignores the media. Use vLLM 0.30+ for media.
⁶ transformers 5 decodes video with torchcodec and falls back to `torchvision.io.read_video`, which torchvision 0.26
removed. Install the torchcodec release that matches your torch (0.11 for torch 2.11).

Differences of 0.01–0.03 in probability are bf16 kernel noise; tokenization is identical. For SGLang,
`--enable-fp32-lm-head` roughly halves the gap. vLLM sends text prompts to `/v1/completions`, and prompts with media
as exact token ids plus the media to `/inference/v1/generate`, so the engine expands the placeholders itself. Models that cannot switch
thinking off need a template that closes the think block.

```bash
python scripts/parity.py --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000
```

### Prompt styles

`--prompt chat` (default) uses the model's chat template, with thinking off. Use it for instruct models, and train
new checkpoints on it too.

`--prompt jevlm` is for checkpoints fine-tuned on a raw completion prompt with no chat template:
`State: … Question: … Options: A. … Answer with the letter of the best option.\nAnswer:`. It is text only.

## JevBench, zero-shot

Stock instruct weights, `--prompt chat`, no fine-tuning, SGLang 0.5.18 for the 27B models and 0.5.9 for the rest, on
A100s. The official JevBench v1.4.0 Score (2026-09-23) needs the 308 sealed items, which only the maintainers run, so
this table has two numbers:

- **Public**: accuracy on the 231 public items (standard 72, easy 48, hard 111), ranked against the 49 entrants'
  published per-item results on the same items.
- **Est. v1.4**: the official `composite_v14` formula (it reproduces every published score). Two inputs are
  assumptions: sealed accuracy is the 25th / 50th / 75th percentile of one-pass entrants whose public accuracy is
  within ±0.05, and sealed-inclusive calibration is C13 × 0.885 (the entrants' median ratio). Cost uses the
  official self-hosted rule (hosted list price of the same weights × 668 measured input tokens per decision).
  Speed uses the official self-hosted adjustment (latency ×2 + 0.15 s). Ranks are among the 71 ranked systems.

| model | public acc | hard 111 | public rank /50 | est. v1.4 score | est. v1.4 rank /72 |
|---|---|---|---|---|---|
| Qwen3.5-27B | 0.879 | 0.775 | #4 | 43.1 / **46.9** / 48.9 | #14 / **#10** / #7 |
| Qwen3.6-27B | 0.866 | 0.748 | #7 | 42.4 / **46.0** / 47.9 | #14 / **#11** / #7 |
| Qwen3.8-27B | 0.840 | 0.703 | #13 | 37.4 / **41.4** / 45.6 | #21 / **#14** / #11 |
| Qwen3.5-9B | 0.810 | 0.667 | #17 | 33.6 / **35.2** / 40.8 | #27 / **#24** / #17 |
| Qwen3.5-4B | 0.740 | 0.595 | #23 | 33.1 / **36.1** / 38.5 | #29 / **#23** / #19 |

The three systems above Qwen3.5-27B on public items are two reasoning LLMs (DeepSeek V4.1 Flash, GPT-5.6 Luna) and
OpenJev in thinking mode. The official v1.4 top five score 54–63. The gap to them comes from the sealed set: one-pass
systems score 0.26–0.36 there (chance is 0.293). For the 27B models the sealed term and the public-to-sealed gap
penalty cut Intelligence from about 83 to about 48, and the Intelligence < 50 gate then lowers the score again.

## Using it for RL

For each question, the readout is a normalized distribution over the options, π(option | state). It comes from one
prefill, on the same engines RL frameworks already use for rollouts, so it can serve as a decision policy directly.
Use the same `llm2jev.prompt.render` and label ids on the training side, so that rollout and learner score the same
logit.

## Tests

```bash
pytest tests                    # fast: fake backend + real tokenizers
LLM2JEV_SLOW=1 pytest tests      # + real models through the transformers backend
```
