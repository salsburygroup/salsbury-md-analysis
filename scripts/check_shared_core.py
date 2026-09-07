"""Compare shared audited contracts without removing experimental additions."""
import argparse
import ast
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--main-source", required=True, type=Path)
p.add_argument("--experimental-source", required=True, type=Path)
args = p.parse_args()
exact_files = (
    "accepted_artifacts.py", "execution_measurement.py", "execution_resources.py",
    "resource_calibrations.py", "memory_policy.py", "convergence.py",
    "rmsd_rg.py", "dccm.py", "coordinate_cache.py", "replica_execution.py",
    "user_workflow.py", "clustering_presentation.py",
)
# These are additive differences, not permission for arbitrary changes elsewhere.
function_exceptions = {
    "resource_planning.py": {"workflow_useful_parallel_cpu_ceiling"},
    "execution_adapters.py": {"_project_cached_modules"},
    "finding_picker.py": {"_report_candidates"},
}
failures, checked = [], []
for name in exact_files:
    a, b = args.main_source / name, args.experimental_source / name
    if not a.is_file() or not b.is_file():
        failures.append(name + ": missing shared file")
    elif ast.dump(ast.parse(a.read_text())) != ast.dump(ast.parse(b.read_text())):
        failures.append(name + ": shared implementation differs")
    checked.append(name)
for name, exceptions in function_exceptions.items():
    def definitions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    a, b = definitions(args.main_source / name), definitions(args.experimental_source / name)
    for function, body in a.items():
        if function in exceptions:
            continue
        checked.append(name + ":" + function)
        if b.get(function) != body:
            failures.append(name + ":" + function)
print(json.dumps({"technical_status": "complete" if not failures else "failed",
                  "shared_contract_count": len(checked), "failures": failures,
                  "explicit_additive_exceptions": {k: sorted(v) for k,v in function_exceptions.items()}},
                 indent=2))
raise SystemExit(bool(failures))
