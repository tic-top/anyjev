"""llm2jev: any chat model, any engine, as a Jev-compatible probability decision service.
One prefill per question, all options in the prompt, one label token read out."""
from .engine import LLM2Jev

__all__ = ["LLM2Jev"]
