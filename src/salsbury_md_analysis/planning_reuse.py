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
        state = {"entries": OrderedDict(), "bytes": 0, "hits": 0, "misses": 0,
                 "schedule_entries": OrderedDict(), "schedule_bytes": 0,
                 "schedule_hits": 0, "schedule_misses": 0, "schedule_evictions": 0}
        token = _active.set(state)
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)
            planning_event("planning_work_reuse_finished", hits=state["hits"],
                           misses=state["misses"], cached_entries=len(state["entries"]),
                           serialized_bytes=state["bytes"],
                           schedule_hits=state["schedule_hits"], schedule_misses=state["schedule_misses"],
                           schedule_evictions=state["schedule_evictions"],
                           schedule_cached_entries=len(state["schedule_entries"]),
                           schedule_serialized_bytes=state["schedule_bytes"])
    return wrapped


def memoized_schedule(function):
    """Reuse exact pure schedule inputs separately from large sampling plans.

    The 128-entry/8-MiB serialized bound applies within one preparation only.
    Every field, dependency, resource limit and node policy enters the key.
    Exceptions are never cached; copies isolate returned owner/token records.
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
        entries = state["schedule_entries"]
        if key in entries:
            state["schedule_hits"] += 1
            entries.move_to_end(key)
            return deepcopy(entries[key][0])
        state["schedule_misses"] += 1
        result = function(*args, **kwargs)
        try:
            size = len(json.dumps(result, separators=(",", ":")).encode())
        except (ValueError, TypeError):
            return result
        if size <= 8 * 1024 * 1024:
            while entries and (len(entries) >= 128 or state["schedule_bytes"] + size > 8 * 1024 * 1024):
                _, (_, old_size) = entries.popitem(last=False)
                state["schedule_bytes"] -= old_size
                state["schedule_evictions"] += 1
            entries[key] = (deepcopy(result), size)
            state["schedule_bytes"] += size
        return result
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
