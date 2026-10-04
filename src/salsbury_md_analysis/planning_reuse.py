"""Bounded exact-input reuse within one preparation, never across runs."""
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json

from .planning_diagnostics import planning_event

_active = ContextVar("planning_work_cache", default=None)
_MAX_ENTRIES = 32
_MAX_SERIALIZED_BYTES = 16 * 1024 * 1024


def reuse_planning_work(function):
    """Share a cache across nested preparation; release it on success or error."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _active.get() is not None:
            return function(*args, **kwargs)
        state = {"entries": OrderedDict(), "bytes": 0, "hits": 0, "misses": 0}
        token = _active.set(state)
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)
            planning_event("planning_work_reuse_finished", hits=state["hits"],
                           misses=state["misses"], cached_entries=len(state["entries"]),
                           serialized_bytes=state["bytes"])
    return wrapped


def memoized_planning(function):
    """Memoize pure plans using every task field and resolved keyword argument.

    Values are deep-copied on storage/readback, so caller annotations cannot
    contaminate later plans. Exceptions and non-JSON inputs are never cached.
    The size bound counts serialized bytes, not a claim about Python heap RSS.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        state = _active.get()
        if state is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            payload = json.dumps(bound.arguments, sort_keys=True, separators=(",", ":"))
        except (ValueError, TypeError):
            return function(*args, **kwargs)
        key = (function.__module__, function.__qualname__, hashlib.sha256(payload.encode()).digest())
        entries = state["entries"]
        if key in entries:
            state["hits"] += 1
            entries.move_to_end(key)
            return deepcopy(entries[key][0])
        state["misses"] += 1
        result = function(*args, **kwargs)
        try:
            size = len(json.dumps(result, separators=(",", ":")).encode())
        except (ValueError, TypeError):
            return result
        if size <= _MAX_SERIALIZED_BYTES:
            while entries and (len(entries) >= _MAX_ENTRIES or state["bytes"] + size > _MAX_SERIALIZED_BYTES):
                _, (_, old_size) = entries.popitem(last=False)
                state["bytes"] -= old_size
            entries[key] = (deepcopy(result), size)
            state["bytes"] += size
        return result
    return wrapped
