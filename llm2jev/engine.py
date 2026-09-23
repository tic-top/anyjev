"""One Jev request -> answers: render once, warm the shared prefix, then hand every question to the engine in one call.

The engine schedules; llm2jev never meters questions itself. A text request is ONE engine call (`score_many`), a
single question runs on the caller's thread, and only media requests and in-process backends use the thread pool."""
from concurrent.futures import ThreadPoolExecutor

from .prompt import find_labels, render
from .scoring import answer, softmax

MAX_QUESTIONS = 64


class LLM2Jev:
    def __init__(self, processor, backend, temperature=1.0, style="chat", workers=16):
        self.processor, self.backend, self.T, self.style = processor, backend, temperature, style
        tok = getattr(processor, "tokenizer", processor)
        # Labels are checked right after the real prompt ending, so a template that ends differently can't
        # silently turn 'A' into two tokens or into ' A'.
        _, probe, _ = render(processor, "x", {"q": {"type": "noul"}}, ["A", "B"], style)
        self.labels, self.ids = find_labels(tok, probe["q"][0])
        self.pool = ThreadPoolExecutor(workers)

    def __call__(self, state, questions):
        return self.run(state, questions)[0]

    def run(self, state, questions):
        """-> (answers, usage). usage.input_tokens counts the prompt tokens the engine processed for the questions
        (prefix-cache hits included, the warm-up request not); output_tokens is the one label position per question."""
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"1..{MAX_QUESTIONS} questions, got {len(questions)}")
        prefix, prompts, images = render(self.processor, state, questions, self.labels, self.style)
        items = list(prompts.items())
        if len(items) == 1:  # on the caller's thread: the shared pool must not cap concurrent requests
            (_, (text, keys)), = items
            row, n = self.backend.score(text, images, self.ids[:len(keys)])
            rows = [row]
        else:
            self.backend.warm(prefix, images)  # branches then hit the prefix cache instead of racing on a cold one
            if not images and hasattr(self.backend, "score_many"):
                k = max(len(keys) for _, (_, keys) in items)  # one label set for the call, sliced per question
                rows, n = self.backend.score_many([text for _, (text, _) in items], self.ids[:k])
            else:
                futs = [self.pool.submit(self.backend.score, text, images, self.ids[:len(keys)]) for _, (text, keys) in items]
                rows, ns = zip(*(f.result() for f in futs))
                n = sum(ns)
        answers = {qid: answer(questions[qid], keys, softmax(row[:len(keys)], self.T))
                   for (qid, (_, keys)), row in zip(items, rows)}
        return answers, {"input_tokens": n, "output_tokens": len(items)}
