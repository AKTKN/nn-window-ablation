"""Process-local timing spans for numerical work and artifact publication."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

_spans: ContextVar[dict[str, float] | None] = ContextVar("training_spans", default=None)


@contextmanager
def capture_spans():
    timings = {}
    token = _spans.set(timings)
    try:
        yield timings
    finally:
        _spans.reset(token)


@contextmanager
def span(name: str):
    timings = _spans.get()
    if timings is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + perf_counter() - started
