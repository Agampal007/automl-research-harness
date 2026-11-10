"""Guarded execution of researcher-written code.

Three layers, none of which is a security boundary on its own:

1. A static AST check: imports must be on an allowlist and file/process/network
   primitives are rejected. This keeps honest-but-sloppy generated code away
   from the held-out labels and the rest of the machine.
2. A separate process with a scrubbed environment, its own working directory,
   and a hard wall-clock timeout that kills the whole process tree.
3. Scoring in the parent: the child writes predictions, it never sees labels.

For adversarial code you would run the child in a container or VM instead.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALLOWED_IMPORTS = {
    "torch", "numpy", "sklearn", "scipy", "lightgbm", "math", "time", "random", "typing",
    "dataclasses", "functools", "itertools", "collections", "copy", "warnings",
    "statistics", "heapq", "bisect", "operator", "abc", "enum", "contextlib",
    "__future__", "gc",
}
BANNED_SUBMODULES = ("sklearn.datasets", "torch.hub", "torch.utils.model_zoo", "torch.multiprocessing")
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
}
BANNED_ATTRS = {
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "dump", "system", "popen", "remove", "rmtree",
    "hub", "load_state_dict_from_url", "__dict__", "__globals__", "__builtins__",
    "__subclasses__", "__class__",
}
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)


def static_check(code: str, entry: str = "fit") -> list[str]:
    """Return a list of problems; empty means the code may be executed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] if node.level == 0 else ["<relative>"]
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in ALLOWED_IMPORTS or m.startswith(BANNED_SUBMODULES):
                problems.append(f"line {node.lineno}: import of '{m}' is not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            problems.append(f"line {node.lineno}: attribute '.{node.attr}' is not allowed")
    has_entry = any(
        isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body
    )
    if not has_entry:
        problems.append(f"missing top-level function `{entry}`")
    return sorted(set(problems))


@dataclass
class RunResult:
    status: str  # ok | crash | timeout | rejected | bad_output
    seconds: float = 0.0
    error: str = ""  # short reason, shown to the researcher
    stdout_tail: str = ""
    stderr_tail: str = ""
    curve: list[dict] = field(default_factory=list)
    preds: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _tail(text: str, n: int = 3000) -> str:
    text = text.strip()
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _spawn(args: list[str], cwd: Path, timeout_s: float) -> tuple[int | None, str, str, float]:
    """Run the trusted runner in a child process. Return code None means timeout."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    out_path, err_path = cwd / "stdout.txt", cwd / "stderr.txt"
    kwargs = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    start = time.time()
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "harness.runner", *args],
            cwd=cwd, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, **kwargs,
        )
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            code = None
        except BaseException:
            _kill_tree(proc)
            raise
    seconds = time.time() - start
    return (
        code,
        out_path.read_text(errors="replace"),
        err_path.read_text(errors="replace"),
        seconds,
    )


def _validate(pred: np.ndarray, n_rows: int, n_classes: int, split: str) -> np.ndarray | str:
    """Normalise predictions to a 1-D integer label array, or describe the problem."""
    if pred.dtype == object or not np.issubdtype(pred.dtype, np.number):
        return f"{split} predictions have non-numeric dtype {pred.dtype}"
    if pred.ndim == 2 and pred.shape == (n_rows, n_classes):
        if not np.isfinite(pred).all():
            return f"{split} predictions contain NaN or inf"
        return pred.argmax(axis=1).astype(np.int64)
    if pred.ndim == 1 and pred.shape[0] == n_rows:
        if not np.isfinite(pred).all():
            return f"{split} predictions contain NaN or inf"
        labels = pred.astype(np.int64)
        if (labels != pred).any() or labels.min() < 0 or labels.max() >= n_classes:
            return f"{split} labels must be integers in [0, {n_classes})"
        return labels
    return (
        f"{split} predictions have shape {pred.shape}; expected ({n_rows},) labels "
        f"or ({n_rows}, {n_classes}) scores"
    )


def run_fit(
    code: str,
    *,
    data_dir: Path,
    workdir: Path,
    seed: int,
    budget_s: float,
    meta: dict,
    grace_s: float = 90.0,
) -> RunResult:
    """Train one candidate with one seed and collect its held-out predictions."""
    problems = static_check(code, "fit")
    if problems:
        return RunResult("rejected", error="Static check failed:\n" + "\n".join(problems))
    workdir = workdir.resolve()  # the child runs with cwd=workdir, so paths must be absolute
    workdir.mkdir(parents=True, exist_ok=True)
    solution = workdir / "solution.py"
    solution.write_text(code, encoding="utf-8")
    code_, out, err, seconds = _spawn(
        ["fit", str(solution), str((data_dir / "public").resolve()), str(workdir.resolve()),
         str(seed), str(budget_s)],
        workdir, budget_s + grace_s,
    )
    res = RunResult("ok", seconds=seconds, stdout_tail=_tail(out), stderr_tail=_tail(err))
    curve_path = workdir / "curve.jsonl"
    if curve_path.exists():
        for line in curve_path.read_text().splitlines():
            try:
                res.curve.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if code_ is None:
        res.status = "timeout"
        res.error = (
            f"Killed after {seconds:.0f}s: exceeded the {budget_s:.0f}s training budget "
            f"plus {grace_s:.0f}s grace. Check elapsed time inside the training loop."
        )
        return res
    if code_ != 0:
        res.status = "crash"
        res.error = f"Process exited with code {code_}.\n{_tail(err, 2500)}"
        return res
    preds_path = workdir / "preds.npz"
    if not preds_path.exists():
        res.status, res.error = "bad_output", "Runner produced no predictions."
        return res
    with np.load(preds_path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    for split in ("val", "test"):
        checked = _validate(raw[split], meta[f"n_{split}"], meta["n_classes"], split)
        if isinstance(checked, str):
            res.status, res.error = "bad_output", checked
            return res
        res.preds[split] = checked
    return res


def run_probe(code: str, *, data_dir: Path, workdir: Path, timeout_s: float = 180.0) -> RunResult:
    """Run a data-inspection script; its stdout is the result."""
    problems = static_check(code, "probe")
    if problems:
        return RunResult("rejected", error="Static check failed:\n" + "\n".join(problems))
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "probe.py"
    script.write_text(code, encoding="utf-8")
    code_, out, err, seconds = _spawn(
        ["probe", str(script), str((data_dir / "public").resolve()), str(workdir.resolve())],
        workdir, timeout_s,
    )
    res = RunResult("ok", seconds=seconds, stdout_tail=_tail(out, 4000), stderr_tail=_tail(err))
    if code_ is None:
        res.status, res.error = "timeout", f"Probe killed after {timeout_s:.0f}s."
    elif code_ != 0:
        res.status, res.error = "crash", f"Probe exited with code {code_}.\n{_tail(err, 2500)}"
    return res
