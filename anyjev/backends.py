"""Backends return the logprobs of chosen token ids at the first generated position, for one prompt.
Every backend does exactly one prefill per prompt and never samples a real answer."""
import base64
import io
import math
import mimetypes
import os

import requests


def _finite(values):
    return [v if v is not None and math.isfinite(v) else -math.inf for v in values]


class SGLang:
    """SGLang /generate: max_new_tokens=1 + token_ids_logprob. Images go as image_data next to the rendered text."""

    def __init__(self, url, timeout=120, **_):
        self.url, self.timeout, self.http = url.rstrip("/"), timeout, requests.Session()

    def _post(self, text, images, ids):
        body = {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                # Every request asks for selected-token logprobs, warm-ups included: SGLang can crash when
                # selected-logprob and plain requests share a batch (sgl-project/sglang#34719).
                "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": ids}
        if images:
            body["image_data"] = images
        r = self.http.post(f"{self.url}/generate", json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        return (out[0] if isinstance(out, list) else out)["meta_info"]

    def warm(self, prefix, images):
        self._post(prefix, images, [0])

    def score(self, text, images, ids):
        rows = (self._post(text, images, ids).get("output_token_ids_logprobs") or [[]])[0]
        got = {int(r[1]): r[0] for r in rows}
        return _finite([got.get(i) for i in ids])


class VLLM:
    """vLLM OpenAI /v1/completions with max_tokens=1 + logprob_token_ids (text prompts).
    Serve with --max-logprobs 256 --return-tokens-as-token-ids. Stock vLLM caps logprob_token_ids per request,
    so large label sets are split into chunks of the SAME prompt (the prefix cache makes repeats cheap).
    Prompts with images go to the tokens-in endpoint /inference/v1/generate (serve with --enable-scale-out): the
    completions API takes no media, and the chat API would re-render the prompt with its own template."""

    def __init__(self, url, model, timeout=120, chunk=128, **_):
        self.url, self.model, self.timeout, self.chunk = url.rstrip("/"), model, timeout, chunk
        self.http = requests.Session()

    def _json(self, path, body):
        r = self.http.post(f"{self.url}{path}", json={"model": self.model, **body}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, text, ids):
        out = self._json("/v1/completions", {"prompt": text, "max_tokens": 1, "temperature": 0.0, "logprobs": 1,
                                             "logprob_token_ids": ids, "add_special_tokens": False})
        top = ((out["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        return [top.get(f"token_id:{i}") for i in ids]

    def _post_media(self, tokens, parts, ids):
        out = self._json("/inference/v1/generate", {"token_ids": tokens, "content_parts": parts, "sampling_params": {
            "max_tokens": 1, "temperature": 0.0, "logprobs": len(ids), "logprob_token_ids": ids}})
        top = (out["choices"][0].get("logprobs") or {"content": [{}]})["content"][0].get("top_logprobs") or []
        got = {t["token"]: t["logprob"] for t in top}
        return [got.get(f"token_id:{i}") for i in ids]

    def _poster(self, text, images):
        if not images:
            return lambda ids: self._post(text, ids)
        tokens = self._json("/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]
        parts = [{"type": "image_url", "url": _uri(s)} for s in images]
        return lambda ids: self._post_media(tokens, parts, ids)

    def warm(self, prefix, images):
        self._poster(prefix, images)([0])

    def score(self, text, images, ids):
        post = self._poster(text, images)
        return _finite([x for i in range(0, len(ids), self.chunk) for x in post(ids[i:i + self.chunk])])


def _uri(src):
    """Local paths become data: URIs, so the engine never needs filesystem access; URLs pass through."""
    if src.startswith(("data:", "http://", "https://")):
        return src
    path = os.path.expanduser(src)
    with open(path, "rb") as f:
        return f"data:{mimetypes.guess_type(path)[0] or 'application/octet-stream'};base64,{base64.b64encode(f.read()).decode()}"


def _load_image(src):
    from PIL import Image
    if src.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1]))).convert("RGB")
    if src.startswith(("http://", "https://")):
        return Image.open(io.BytesIO(requests.get(src, timeout=30).content)).convert("RGB")
    return Image.open(os.path.expanduser(src)).convert("RGB")


class HF:
    """In-process transformers reference: full-vocab log-softmax at the last prompt position. Slow (no prefix cache),
    meant for correctness checks against the engines and for models no engine supports yet."""

    def __init__(self, model, device=None, dtype="bfloat16", **_):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = AutoConfig.from_pretrained(model)
        multimodal = hasattr(cfg, "vision_config")
        cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        self.model = cls.from_pretrained(model, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model) if multimodal else None
        self.tok = AutoTokenizer.from_pretrained(model)

    def warm(self, prefix, images):
        pass

    def score(self, text, images, ids):
        torch = self.torch
        if images:
            if self.processor is None:
                raise ValueError("this model has no vision tower")
            inputs = self.processor(text=[text], images=[_load_image(s) for s in images], return_tensors="pt")
        else:
            inputs = {"input_ids": torch.tensor([self.tok.encode(text, add_special_tokens=False)])}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float().log_softmax(-1)
        return [float(logits[i]) for i in ids]


BACKENDS = {"sglang": SGLang, "vllm": VLLM, "hf": HF}
