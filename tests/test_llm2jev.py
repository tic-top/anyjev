"""Fast tests use a fake backend and a real tokenizer. LLM2JEV_SLOW=1 also runs real models through the HF backend
(CPU is fine): Qwen3-0.6B for text, Qwen3.5-2B for image and video questions, Qwen2-Audio-7B for audio."""
import math
import os

import pytest
from transformers import AutoTokenizer

from llm2jev import LLM2Jev
from llm2jev.backends import HF
from llm2jev.prompt import render, state_messages

TEXT_MODEL, VL_MODEL, AUDIO_MODEL = "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-2B", "Qwen/Qwen2-Audio-7B-Instruct"
slow = pytest.mark.skipif(not os.environ.get("LLM2JEV_SLOW"), reason="set LLM2JEV_SLOW=1 to run real models")
STATE = [{"role": "system", "content": "You are a support assistant."},
         {"role": "user", "content": "I was charged twice. Please refund the duplicate."}]
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["Routine", "Urgent", "Emergency"]},
}


class Fake:
    """Returns fixed logprobs; records what it was asked."""

    def __init__(self):
        self.calls, self.warms = [], []

    def warm(self, prefix, images):
        self.warms.append(prefix)

    def score(self, text, images, ids):
        self.calls.append((text, ids))
        return [-float(i) for i in range(len(ids))], 10


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TEXT_MODEL)


def test_labels_are_single_tokens_after_answer(tok):
    jev = LLM2Jev(tok, Fake())
    assert len(jev.labels) == 255 and jev.labels[:3] == ["A", "B", "C"] and len(set(jev.ids)) == 255
    _, prompts, _ = render(tok, STATE, QUESTIONS, jev.labels)
    for text, _ in prompts.values():
        base = tok.encode(text, add_special_tokens=False)
        for label, i in zip(jev.labels[:40], jev.ids):
            assert tok.encode(text + " " + label, add_special_tokens=False) == base + [i]


def test_prefix_is_shared_token_for_token(tok):
    jev = LLM2Jev(tok, Fake())
    prefix, prompts, _ = render(tok, STATE, QUESTIONS, jev.labels)
    seqs = [tok.encode(t, add_special_tokens=False) for t, _ in prompts.values()]
    common = min(len(os.path.commonprefix([seqs[0], s])) for s in seqs)
    assert all(t.startswith(prefix) for t, _ in prompts.values())
    assert common >= len(tok.encode(prefix, add_special_tokens=False)) - 1  # at most the boundary token re-merges


def test_answers_and_warmup(tok):
    fake = Fake()
    out = LLM2Jev(tok, fake)(STATE, QUESTIONS)
    p = [math.exp(-i) for i in range(3)]
    p = [x / sum(p) for x in p]
    assert len(fake.warms) == 1 and len(fake.calls) == 3
    assert out["refund"]["noul"] == pytest.approx(1 / (1 + math.exp(-1)))  # Yes is option A
    assert out["department"]["choice"] == "billing"
    assert out["urgency"]["score"] == pytest.approx(p[1] + 2 * p[2])
    assert all("Options:\nA. " in t for t, _ in fake.calls)


def test_image_parts_become_placeholders(tok):
    state = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}},
                                          {"type": "text", "text": "look"}]}]
    proc = AutoTokenizer.from_pretrained(VL_MODEL)
    _, prompts, images = render(proc, state, {"q": {"type": "noul", "instructions": "Red?"}}, ["A", "B"])
    assert images == ["data:x"] and prompts["q"][0].count("<|image_pad|>") == 1


def test_video_and_audio_parts_keep_their_kind():
    state = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "v.mp4"}},
                                          {"type": "image", "image": "i.png"}, {"type": "audio", "audio": "a.wav"}]}]
    msgs, media = state_messages(state)
    assert media == [("video", "v.mp4"), "i.png", ("audio", "a.wav")]
    assert [p["type"] for p in msgs[0]["content"]] == ["video", "image", "audio"]


def test_jevlm_style_is_the_raw_letters_prompt(tok):
    jev = LLM2Jev(tok, Fake(), style="jevlm")
    _, prompts, _ = render(tok, "The sky is blue.", QUESTIONS, jev.labels, "jevlm")
    assert prompts["refund"][0] == ("State:\nThe sky is blue.\n\nQuestion: Does the user request a refund?\nOptions:\n"
                                    "A. false: No\nB. true: Yes\nAnswer with the letter of the best option.\nAnswer:")
    assert prompts["refund"][1] == ["false", "true"] and jev.labels[:2] == ["A", "B"]
    assert "A. 0: Routine\nB. 1: Urgent\nC. 2: Emergency\n" in prompts["urgency"][0]


def _question(instructions, *options):
    return {"q": {"type": "choice", "instructions": instructions, "criteria": dict.fromkeys(options)}}


def _cpu_friendly():
    """Qwen3.5 linear-attention layers default to CUDA-only kernels; use transformers' torch fallbacks on CPU."""
    import torch
    if torch.cuda.is_available():
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        f = getattr(m, name)
        setattr(m, name, getattr(f, "__wrapped__", f))


@slow
def test_hf_text_sees_all_options(tok):
    jev = LLM2Jev(tok, HF(TEXT_MODEL, dtype="float32"))
    q = lambda opts: {"q": {"type": "choice", "instructions": "What color is the sky?", "criteria": dict.fromkeys(opts)}}
    _, a, _ = render(tok, "The sky is blue.", q(["red", "green", "none of the above"]), jev.labels)
    _, b, _ = render(tok, "The sky is blue.", q(["red", "blue", "none of the above"]), jev.labels)
    la = jev.backend.score(a["q"][0], [], jev.ids[:3])[0]
    lb = jev.backend.score(b["q"][0], [], jev.ids[:3])[0]
    assert abs(la[0] - lb[0]) > 1e-3  # option A's raw logit moved when only option B changed
    # Qwen3-0.6B zero-shot prefers "none of the above" here in every prompt ending tried (labels hold ~99% of the
    # next-token mass), so the accuracy check uses plain options: a model limit, not a readout bug.
    assert jev("The sky is blue.", q(["red", "blue", "green"]))["q"]["choice"] == "blue"
    out = jev(STATE, QUESTIONS)
    assert set(out) == set(QUESTIONS) and out["department"]["choice"] == "billing"


@slow
def test_mlx_matches_hf(tok):
    pytest.importorskip("mlx_lm")
    from llm2jev.backends import MLX
    got = LLM2Jev(tok, MLX(TEXT_MODEL))(STATE, QUESTIONS)
    ref = LLM2Jev(tok, HF(TEXT_MODEL, dtype="float32"))(STATE, QUESTIONS)
    assert got["department"]["choice"] == ref["department"]["choice"]
    assert got["refund"]["noul"] == pytest.approx(ref["refund"]["noul"], abs=0.03)


@slow
def test_hf_image_question(tmp_path):
    from PIL import Image
    from transformers import AutoProcessor
    _cpu_friendly()
    img = tmp_path / "red.png"
    Image.new("RGB", (64, 64), (220, 20, 20)).save(img)
    proc = AutoProcessor.from_pretrained(VL_MODEL)
    jev = LLM2Jev(proc, HF(VL_MODEL, dtype="float32"))
    state = [{"role": "user", "content": [{"type": "image", "image": str(img)}, {"type": "text", "text": "Here is a picture."}]}]
    out = jev(state, {"color": {"type": "choice", "instructions": "What color is the square?",
                                "criteria": {"red": None, "blue": None, "green": None}}})
    assert out["color"]["choice"] == "red", out


@slow
def test_hf_video_question(tmp_path):
    import sys
    from transformers import AutoProcessor
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from parity import moving_square
    _cpu_friendly()
    moving_square(str(tmp_path / "move.mp4"))
    jev = LLM2Jev(AutoProcessor.from_pretrained(VL_MODEL), HF(VL_MODEL, dtype="float32"))
    state = [{"role": "user", "content": [{"type": "video", "video": str(tmp_path / "move.mp4")}]}]
    out = jev(state, _question("Which way does the red square move?", "left", "right", "up", "down"))
    assert out["q"]["choice"] == "right", out


@slow
def test_hf_audio_question(tmp_path):
    import sys
    from transformers import AutoProcessor
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from parity import beep
    beep(str(tmp_path / "beep.wav"))
    jev = LLM2Jev(AutoProcessor.from_pretrained(AUDIO_MODEL), HF(AUDIO_MODEL))
    state = [{"role": "user", "content": [{"type": "audio", "audio": str(tmp_path / "beep.wav")}]}]
    out = jev(state, _question("What is in the recording?", "a person talking", "a steady electronic beep", "a dog barking"))
    assert out["q"]["choice"] == "a steady electronic beep", out


def test_backend_400_is_422_and_outage_is_504(tok):
    import json
    import threading
    import urllib.error
    import urllib.request
    import requests
    from llm2jev.__main__ import Handler, Server

    class Failing(Fake):
        def score(self, text, images, ids):
            r = requests.Response()
            r.status_code, r._content = self.code, b"prompt too long"
            r.raise_for_status()

    def post(code):
        fake = Failing()
        fake.code = code
        srv = Server(("127.0.0.1", 0), Handler)
        srv.jev, srv.name = LLM2Jev(tok, fake), "m"
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        body = json.dumps({"state": "x", "questions": {"q": {"type": "noul"}}}).encode()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/v1/systemone", body)
        except urllib.error.HTTPError as e:
            return e.code
        finally:
            srv.shutdown()

    assert post(400) == 422 and post(503) == 504


def test_pre_060_backend_returning_bare_list_gets_clear_error(tok):
    class Old(Fake):
        def score(self, text, images, ids):
            return super().score(text, images, ids)[0]
    jev = LLM2Jev(tok, Old())
    for questions in ({"q": {"type": "noul"}}, QUESTIONS):  # 2 floats would unpack silently; single + pooled paths
        with pytest.raises(RuntimeError, match=r"\(logprobs, prompt_tokens\) tuple"):
            jev(STATE, questions)


def test_strict_alternation_template_gets_question_in_last_user_turn(tok):
    class Strict:  # Gemma-style template: consecutive user turns raise
        def apply_chat_template(self, msgs, **kw):
            if any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
                raise ValueError("Conversation roles must alternate")
            return tok.apply_chat_template(msgs, **kw)
    prefix, prompts, _ = render(Strict(), STATE, QUESTIONS, ["A", "B", "C"])
    text = prompts["refund"][0]
    assert text.startswith(prefix) and text.count("<|im_start|>user") == 1
    assert "refund the duplicate.\n\nEvaluate the conversation" in text


def test_batched_backend_gets_one_call_and_same_answers(tok):
    class Batch(Fake):
        def score_many(self, texts, ids):
            self.batches = getattr(self, "batches", 0) + 1
            return [self.score(t, [], ids)[0] for t in texts], 10 * len(texts)
    plain, batch = Fake(), Batch()
    assert LLM2Jev(tok, batch)(STATE, QUESTIONS) == LLM2Jev(tok, plain)(STATE, QUESTIONS)
    assert batch.batches == 1 and len(batch.warms) == 1
    assert {len(ids) for _, ids in batch.calls} == {3}  # the batch asks for the widest label set, sliced per question


def test_usage_counts_engine_prompt_tokens(tok):
    jev = LLM2Jev(tok, Fake())
    assert jev.run(STATE, QUESTIONS)[1] == {"input_tokens": 30, "output_tokens": 3}
    assert jev.run(STATE, {"refund": QUESTIONS["refund"]})[1] == {"input_tokens": 10, "output_tokens": 1}


def test_server_takes_a_burst_of_connections():
    from llm2jev.__main__ import Server
    assert Server.request_queue_size >= 256  # the stdlib default of 5 reset connections at 64 concurrent clients
