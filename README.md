# AnyJev

Turn **any chat model** into a [Jev](https://docs.typesafe.ai/api)-compatible probability decision service, on
**SGLang, vLLM or plain transformers**, with no model or engine changes.

```
state + question + ALL options  ──one prefill──▶  logprobs of the option labels at one position  ──▶  probabilities
```

- **One forward per question**, however many options (up to 255). No generation, no sampling.
- **All options in one prompt**, so options compete directly ("none of the above" can depend on the others).
- **Standard chat template, thinking off.** Works with instruct models zero-shot; fine-tuned checkpoints use the same prompt.
- **Shared prefix, rendered once.** Every question of a request starts with byte-identical text, the prefix is warmed
  once, and the engine's prefix cache (SGLang radix / vLLM APC) serves the rest.
- **Images in the state** (SGLang, vLLM and transformers backends) for multimodal models.

AnyJev is an independent implementation of the documented System One wire format. It is not affiliated with TypeSafe.

## Quick start

```bash
pip install "anyjev @ git+https://github.com/tic-top/anyjev"          # client + server; the engine runs separately
pip install "anyjev[hf,vision] @ git+https://github.com/tic-top/anyjev" # + in-process transformers backend

# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3.5-2B --port 30000
anyjev --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000

# vLLM (--enable-scale-out opens the tokens-in endpoint used for images)
vllm serve Qwen/Qwen3.5-2B --max-logprobs 256 --return-tokens-as-token-ids --enable-scale-out --port 8000
anyjev --model Qwen/Qwen3.5-2B --backend vllm --url http://127.0.0.1:8000

# transformers, in-process reference (slow, no prefix cache)
anyjev --model Qwen/Qwen3-0.6B --backend hf
```

From Python, without the HTTP server:

```python
from transformers import AutoProcessor
from anyjev import AnyJev
from anyjev.backends import SGLang

jev = AnyJev(AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B"), SGLang("http://127.0.0.1:30000"))
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

Images: put `{"type": "image", "image": "<path | https URL | data: URI>"}` (or OpenAI-style `image_url`) parts in a
state message.

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

| backend | text | image |
|---|---|---|
| transformers | reference (Qwen3-0.6B, Qwen3.5-2B) | reference (Qwen3.5-2B) |
| SGLang 0.5.9 | parity ✓ Qwen3-0.6B, Qwen3.5-2B | parity ✓ Qwen3.5-2B |
| vLLM 0.30.0 | parity ✓ Qwen3-0.6B, Qwen3.5-2B | parity ✓ Qwen3.5-2B |

Differences of 0.01–0.03 in probability are bf16 kernel noise; tokenization is identical. For SGLang,
`--enable-fp32-lm-head` roughly halves the gap. vLLM sends text prompts to `/v1/completions`, and prompts with images
as exact token ids plus the images to `/inference/v1/generate`, so both engines score the same bytes. Audio and video parts are not supported yet. Models that cannot switch
thinking off need a template that closes the think block.

```bash
python scripts/parity.py --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000
```

### Prompt styles

`--prompt chat` (default) uses the model's chat template, with thinking off. Use it for instruct models, and train
new checkpoints on it too.

`--prompt jevlm` is for checkpoints fine-tuned on a raw completion prompt with no chat template:
`State: … Question: … Options: A. … Answer with the letter of the best option.\nAnswer:`. It is text only.

## Using it for RL

For each question, the readout is a normalized distribution over the options, π(option | state). It comes from one
prefill, on the same engines RL frameworks already use for rollouts, so it can serve as a decision policy directly.
Use the same `anyjev.prompt.render` and label ids on the training side, so that rollout and learner score the same
logit.

## Tests

```bash
pytest tests                    # fast: fake backend + real tokenizers
ANYJEV_SLOW=1 pytest tests      # + real models through the transformers backend
```
