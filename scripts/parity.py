"""Check an engine backend against the in-process transformers reference, label by label.

    python scripts/parity.py --model Qwen/Qwen3-0.6B --backend sglang --url http://127.0.0.1:30000

For every question of a few built-in requests, both sides score the SAME rendered prompt; the script prints the max
|logprob difference| over the question's labels, the probability gap after softmax, and whether the argmax agrees.
Exit code 1 if any argmax disagrees or a probability differs by more than --tol.
"""
import argparse
import base64
import io
import math
import os
import struct
import sys
import tempfile
import time
import wave

from llm2jev.backends import BACKENDS, HF
from llm2jev.prompt import find_labels, render
from llm2jev.scoring import softmax

STATE = [{"role": "system", "content": "You are a support assistant."},
         {"role": "user", "content": "I was charged twice. Please refund the duplicate."}]
LONG = "\n".join(f"Log line {i}: user {i % 7} opened ticket #{1000 + i} about invoice {i * 37 % 101}." for i in range(200))
COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "India", "Norway", "Chile", "Egypt", "Vietnam", "Peru",
             "Spain", "Italy", "Mexico", "Ghana", "Nepal", "Poland", "Greece", "Turkey", "Iran", "Iraq", "Cuba",
             "Chad", "Mali", "Laos", "Oman", "Fiji", "Togo", "Niger", "Qatar"]


def moving_square(path, n=16, size=320, fps=8):
    """A red square sliding left to right on white, written as H.264 mp4 (needs PyAV). 320px: engines upscale smaller
    videos differently (SGLang raises 128px frames to ~320px, HF and vLLM keep them)."""
    import av
    import numpy as np
    out = av.open(path, "w")
    stream = out.add_stream("libx264", rate=fps)
    stream.width = stream.height = size
    stream.pix_fmt = "yuv420p"
    for i in range(n):
        frame = np.full((size, size, 3), 255, np.uint8)
        x = 4 + i * (size - size // 4 - 8) // (n - 1)
        frame[size // 3:size // 3 + size // 4, x:x + size // 4] = (220, 20, 20)
        out.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
    out.mux(stream.encode())
    out.close()


def beep(path, hz=440, seconds=2, rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * hz * t / rate)))
                               for t in range(seconds * rate)))


def requests_for(vision, audio=False):
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
        video = os.path.join(tempfile.mkdtemp(), "move.mp4")
        moving_square(video)
        out.append(("video", [{"role": "user", "content": [{"type": "video", "video": video},
                                                           {"type": "text", "text": "Here is a video."}]}],
                    {"direction": {"type": "choice", "instructions": "Which way does the red square move?",
                                   "criteria": {"left": None, "right": None, "up": None, "down": None}}}))
    if audio:  # a short clip and a full 30 s Whisper window: SGLang 0.5.9 only matches on the full window
        for seconds in (2, 30):
            clip = os.path.join(tempfile.mkdtemp(), "beep.wav")
            beep(clip, seconds=seconds)
            out.append((f"audio-{seconds}s", [{"role": "user", "content": [{"type": "audio", "audio": clip},
                                                                          {"type": "text", "text": "Here is a recording."}]}],
                        {"sound": {"type": "choice", "instructions": "What is in the recording?",
                                   "criteria": {"speech": "a person talking", "tone": "a steady electronic beep",
                                                "dog": "a dog barking"}}}))
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
    cfg = ref.model.config
    for name, state, questions in requests_for(hasattr(cfg, "vision_config"), hasattr(cfg, "audio_config")):
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
