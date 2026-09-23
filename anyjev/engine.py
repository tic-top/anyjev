"""One Jev request -> answers: render once, warm the shared prefix, then score every question concurrently."""
from concurrent.futures import ThreadPoolExecutor

from .prompt import find_labels, render
from .scoring import answer, softmax

MAX_QUESTIONS = 64


class AnyJev:
    def __init__(self, processor, backend, temperature=1.0, style="chat", workers=16):
        self.processor, self.backend, self.T, self.style = processor, backend, temperature, style
        tok = getattr(processor, "tokenizer", processor)
        # Labels are checked right after the real prompt ending, so a template that ends differently can't
        # silently turn 'A' into two tokens or into ' A'.
        _, probe, _ = render(processor, "x", {"q": {"type": "noul"}}, ["A", "B"], style)
        self.labels, self.ids = find_labels(tok, probe["q"][0])
        self.pool = ThreadPoolExecutor(workers)

    def __call__(self, state, questions):
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"1..{MAX_QUESTIONS} questions, got {len(questions)}")
        prefix, prompts, images = render(self.processor, state, questions, self.labels, self.style)
        if len(prompts) > 1:
            self.backend.warm(prefix, images)  # later branches hit the prefix cache instead of racing on a cold one
        jobs = {qid: (keys, self.pool.submit(self.backend.score, text, images, self.ids[:len(keys)]))
                for qid, (text, keys) in prompts.items()}
        return {qid: answer(questions[qid], keys, softmax(fut.result(), self.T)) for qid, (keys, fut) in jobs.items()}

