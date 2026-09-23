"""Check an engine backend against the in-process transformers reference, label by label.

    python scripts/parity.py --model Qwen/Qwen3-0.6B --backend sglang --url http://127.0.0.1:30000

For every question of a few built-in requests, both sides score the SAME rendered prompt; the script prints the max
|logprob difference| over the question's labels, the probability gap after softmax, and whether the argmax agrees.
Exit code 1 if any argmax disagrees or a probability differs by more than --tol.
"""
import argparse
import base64
import io
import sys
import time

from anyjev.backends import BACKENDS, HF
from anyjev.prompt import find_labels, render
from anyjev.scoring import softmax

STATE = [{"role": "system", "content": "You are a support assistant."},
         {"role": "user", "content": "I was charged twice. Please refund the duplicate."}]
LONG = "\n".join(f"Log line {i}: user {i % 7} opened ticket #{1000 + i} about invoice {i * 37 % 101}." for i in range(200))
COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "India", "Norway", "Chile", "Egypt", "Vietnam", "Peru",
             "Spain", "Italy", "Mexico", "Ghana", "Nepal", "Poland", "Greece", "Turkey", "Iran", "Iraq", "Cuba",
             "Chad", "Mali", "Laos", "Oman", "Fiji", "Togo", "Niger", "Qatar"]


def requests_for(vision):
    out = [
        ("support", STATE, {
            "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
            "department": {"type": "choice", "instructions": "Which department should handle this?",
                           "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
            "urgency": {"type": "score", "instructions": "How urgent is the request?",
                        "criteria": ["Routine", "Urgent", "Emergency"]}}),
        ("30-way", "Tokyo is the capital city.", {
            "country": {"type": "choice", "instructions": "Which country is this city the capital of?",
                        "criteria": dict.fromkeys(COUNTRIES)}}),
        ("long-state", LONG, {
            "invoice": {"type": "noul", "instructions": "Does any log line mention invoice 0?"},
            "user": {"type": "choice", "instructions": "Which user opened ticket #1003?",
                     "criteria": {f"user {i}": None for i in range(7)}}}),
    ]
    if vision:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        out.append(("image", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                           {"type": "text", "text": "Here is a picture."}]}],
                    {"color": {"type": "choice", "instructions": "What color is the square?",
                               "criteria": {"red": None, "blue": None, "green": None}}}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--served-model-name")
    ap.add_argument("--device", help="device for the HF reference (default: cuda if available)")
    ap.add_argument("--dtype", default="bfloat16", help="HF reference dtype; float32 separates engine noise from reference noise")
    ap.add_argument("--tol", type=float, default=0.03,
                    help="max allowed probability difference; bf16 engine kernels differ from HF by up to ~0.03")
    a = ap.parse_args()

    ref = HF(a.model, device=a.device, dtype=a.dtype)
    processor = ref.processor or ref.tok
    engine = BACKENDS[a.backend](url=a.url, model=a.served_model_name or a.model)
    _, probe, _ = render(processor, "x", {"q": {"type": "noul"}}, ["A", "B"])
    labels, ids = find_labels(ref.tok, probe["q"][0])

    bad = 0
    print(f"{'request':12} {'question':10} {'K':>3} {'max|dℓ|':>8} {'max|dp|':>8} {'argmax':>7} {'engine ms':>9}")
    for name, state, questions in requests_for(ref.processor is not None):
        prefix, prompts, images = render(processor, state, questions, labels)
        engine.warm(prefix, images)
        for qid, (text, keys) in prompts.items():
            k = ids[:len(keys)]
            t0 = time.perf_counter()
            le = engine.score(text, images, k)
            ms = (time.perf_counter() - t0) * 1e3
            lr = ref.score(text, images, k)
            pe, pr = softmax(le), softmax(lr)
            dl = max(abs(x - y) for x, y in zip(le, lr))
            dp = max(abs(x - y) for x, y in zip(pe, pr))
            same = max(range(len(pe)), key=pe.__getitem__) == max(range(len(pr)), key=pr.__getitem__)
            bad += (not same) or dp > a.tol
            print(f"{name:12} {qid:10} {len(k):3d} {dl:8.4f} {dp:8.4f} {'ok' if same else 'DIFF':>7} {ms:9.1f}")
    print("PASS" if not bad else f"FAIL ({bad} questions)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
