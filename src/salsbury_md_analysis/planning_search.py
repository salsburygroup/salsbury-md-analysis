"""Deterministic bounds on optional planning refinement, not scientific work."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from .planning_diagnostics import planning_event

DEFAULT_REFINEMENT_SCHEDULE_CALLS = 512
_active = ContextVar("planning_search_scope", default=None)


class RefinementBudgetExceeded(RuntimeError):
    """Extra-sampling search stopped; minimum feasibility is not rejected."""


@contextmanager
def search_scope(limit=DEFAULT_REFINEMENT_SCHEDULE_CALLS):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("maximum_refinement_schedule_calls must be a nonnegative integer")
    if _active.get() is not None:
        yield _active.get()
        return
    state = {"limit": limit, "refinement_calls": 0, "minimum_calls": 0,
             "phase": "minimum_feasibility", "fallbacks": 0,
             "inner_schedule_cache_hits": 0, "inner_schedule_cache_misses": 0}
    token = _active.set(state)
    try:
        yield state
    finally:
        planning_event("planning_search_finished", **state)
        _active.reset(token)


def bounded_campaign_search(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        limit = kwargs["analysis_config"].get("planning", {}).get(
            "maximum_refinement_schedule_calls", DEFAULT_REFINEMENT_SCHEDULE_CALLS)
        with search_scope(limit):
            return function(*args, **kwargs)
    return wrapped


def record_schedule_work():
    """Charge only actual schedule constructions; cache hits cost no calls."""
    state = _active.get()
    if state is None:
        return
    if state["phase"] == "refinement":
        if state["refinement_calls"] >= state["limit"]:
            raise RefinementBudgetExceeded(
                f"optional refinement reached {state['limit']} native schedule calls")
        state["refinement_calls"] += 1
    else:
        state["minimum_calls"] += 1
    total = state["refinement_calls"] + state["minimum_calls"]
    if total == 1 or total % 128 == 0:
        planning_event("planning_schedule_progress", **state)


def record_inner_schedule_cache(before, after):
    state = _active.get()
    if state is not None:
        state["inner_schedule_cache_hits"] += after.hits - before.hits
        state["inner_schedule_cache_misses"] += after.misses - before.misses


def minimum_then_refine(planner, tasks, *, minimum_only=False,
                        verify_rejected_minimum=False, _minimum_plan=None, **kwargs):
    """Preserve a complete minimum candidate before attempting extra sampling.

    Final native preparation still validates this candidate before emitting a
    launcher. An interrupted refinement never proves the requested scope cannot
    fit. Unknown input errors are not caught or converted into a fallback.
    Internal callers may pass a minimum they just computed for identical tasks
    and options; this avoids repeating the same feasibility calculation.
    """
    from .resource_planning import PlanningSearchError

    with search_scope() as state:
        previous = state["phase"]
        state["phase"] = "minimum_feasibility"
        try:
            minimum = (deepcopy(_minimum_plan) if _minimum_plan is not None
                       else planner(tasks, _minimum_only=True, **kwargs))
            minimum_fits = minimum.get("feasibility_status") == "feasible"
            if minimum_only or (not minimum_fits and not verify_rejected_minimum):
                return minimum
            planning_event("planning_minimum_checked", planner=getattr(planner, "__name__", type(planner).__name__),
                           task_count=len(tasks), feasible=minimum_fits)
            state["phase"] = "refinement"
            reason = None
            try:
                if state["refinement_calls"] >= state["limit"]:
                    raise RefinementBudgetExceeded("optional refinement budget exhausted")
                refined = planner(tasks, _minimum_only=False, **kwargs)
                if refined.get("feasibility_status") == "feasible":
                    plan = refined
                elif not minimum_fits:
                    return refined
                else:
                    reason = "refined candidate did not validate; minimum candidate retained"
            except (RefinementBudgetExceeded, PlanningSearchError) as exc:
                reason = str(exc)
            if reason is not None:
                if not minimum_fits:
                    raise PlanningSearchError(reason, diagnostics={
                        "minimum_candidate_feasible": False,
                        "refinement_schedule_calls": state["refinement_calls"],
                        "search_budget_exhausted": state["refinement_calls"] >= state["limit"],
                    })
                plan = deepcopy(minimum)
                state["fallbacks"] += 1
                planning_event("planning_refinement_fallback", reason=reason, **state)
            plan["planning_refinement"] = {
                "status": "validated_minimum_fallback" if reason else "completed",
                "reason": reason, "minimum_checked_first": True,
                "maximum_schedule_calls": state["limit"],
                "used_schedule_calls": state["refinement_calls"],
                "optimality_proven": False,
                "scientific_contracts_unchanged": True,
            }
            return plan
        finally:
            state["phase"] = previous
