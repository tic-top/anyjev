"""Old import name, kept so `import anyjev` / `from anyjev.backends import ...` keep working. Use `llm2jev`."""
import importlib
import sys

from llm2jev import *  # noqa: F401,F403
from llm2jev import AnyJev, LLM2Jev  # noqa: F401

for _m in ("backends", "engine", "prompt", "scoring"):
    sys.modules[f"{__name__}.{_m}"] = importlib.import_module(f"llm2jev.{_m}")
