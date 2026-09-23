"""python -m anyjev --model Qwen/Qwen3.5-2B --backend sglang --url http://127.0.0.1:30000

    POST /v1/systemone  {model?, state, questions: {id: {type, instructions, criteria}}} -> {id, model, answers}
    GET  /v1/models     GET /health
"""
import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .backends import BACKENDS
from .engine import AnyJev


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        name = self.server.name
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok", "model": name, "temperature": self.server.jev.T})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "anyjev"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            answers = self.server.jev(body.get("state", ""), body.get("questions") or {})
        except (ValueError, KeyError, TypeError, NotImplementedError) as exc:
            return self._send(422, {"error": str(exc)})
        except requests.HTTPError as exc:  # backend 400 = a request it cannot hold (e.g. over the context window): the client's problem
            code = 422 if exc.response is not None and exc.response.status_code == 400 else 504
            return self._send(code, {"error": f"backend: {exc.response.text if exc.response is not None else exc}"})
        except requests.RequestException as exc:
            return self._send(504, {"error": f"backend: {exc}"})
        self._send(200, {"id": f"jev-{uuid.uuid4().hex[:16]}", "model": body.get("model") or self.server.name,
                         "answers": answers})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or path; supplies the chat template and tokenizer")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="sglang")
    ap.add_argument("--url", default="http://127.0.0.1:30000", help="engine base URL (sglang / vllm)")
    ap.add_argument("--served-model-name", help="model name the vLLM server was started with (default: --model)")
    ap.add_argument("--temperature", type=float, default=1.0, help="softmax temperature over label logprobs")
    ap.add_argument("--prompt", choices=["chat", "jevlm"], default="chat",
                    help="chat: model chat template, thinking off. jevlm: raw completion prompt for checkpoints trained without a chat template")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(a.model)
        processor.apply_chat_template  # plain tokenizers come back from AutoProcessor too
    except (OSError, ValueError, AttributeError):
        processor = AutoTokenizer.from_pretrained(a.model)
    backend = BACKENDS[a.backend](url=a.url, model=a.served_model_name or a.model)
    serve(AnyJev(processor, backend, a.temperature, a.prompt), a.served_model_name or a.model, a.host, a.port)


def serve(jev, name, host="127.0.0.1", port=8080):
    """Blocking /v1/systemone server around an AnyJev instance (also used by projects that build their own AnyJev)."""
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.jev, srv.name = jev, name
    print(f"anyjev on http://{host}:{port}  model={name} prompt={jev.style} T={jev.T:.4f} labels={len(jev.labels)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
