"""llm2jev: any chat model, any engine, as a Jev-compatible probability decision service.
One prefill per question, all options in the prompt, one label token read out."""
from .engine import LLM2Jev

AnyJev = LLM2Jev  # name before the rename; the `anyjev` package re-exports everything

__all__ = ["LLM2Jev", "AnyJev"]
