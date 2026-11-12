"""Prompt construction and reply parsing for the researcher model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

KINDS = ("architecture", "loss", "features", "optimization", "regularization", "ablation", "hparam")

SYSTEM = """\
You are the ML researcher in an automated experimentation loop. Nobody reviews \
your work between iterations: each reply you write is parsed by a program, your \
code is executed, and the measured result comes back to you on the next turn. \
The goal is the highest validation accuracy you can reach and defend on the \
dataset below.

# Dataset
{profile}

# How experiments run
Each experiment is one complete Python file that replaces the current best \
solution ("the incumbent"). It must define:

    def fit(X_train, y_train, *, seed, device, time_budget_s, log):
        ...
        return predict          # predict(X: np.ndarray) -> np.ndarray

- X_train is a float32 array of raw, unscaled features; y_train is int64 labels. \
Validation and test rows are never passed to fit. If you want early stopping or \
model selection, split it off X_train yourself.
- predict receives raw features with the same columns and returns either class \
labels of shape (n,) or class scores of shape (n, n_classes). It is called on \
several hundred thousand rows at once, so batch the forward pass.
- device is "cuda" or "cpu". Training must finish within time_budget_s seconds \
of wall-clock time (currently {budget:.0f}); a run that overshoots by much is \
killed and counts as a failure, so check elapsed time inside the loop. The GPU \
is shared: keep memory under about 2 GB.
- Call log(step, train_loss=..., holdout_acc=...) with whatever scalars you want \
to see as a curve next turn. print() output is returned to you as well.
- Allowed imports: torch, numpy, sklearn, scipy and the standard-library basics \
(math, time, random, typing, dataclasses, functools, itertools, collections, \
copy, warnings). No file, network, process or environment access.

# How results are judged
The harness scores your predictions on a held-out validation split. A candidate \
is first run with one seed; if that beats the incumbent's mean it is re-run with \
more seeds and becomes the new incumbent only if its mean gain is larger than \
the seed-to-seed noise. Everything else is discarded and the incumbent stays. A \
separate test split is scored silently for the final report, so tuning to \
validation noise will show up there.
{reference}

# What makes a good experiment
Change one idea at a time relative to the incumbent, so the result tells you \
something. State the hypothesis so that the result can prove it wrong. Use the \
experiment log: do not repeat an idea that already failed unless you can say \
what is different this time, and when several small changes have been accepted, \
consider an ablation that removes one to check it still earns its place. \
Architectures, feature construction, loss functions, optimisation and \
regularisation are all in scope; so are non-neural models if you think they win.

You may also run a probe instead of an experiment: a file defining \
`probe(X_train, y_train)` that prints whatever you want to learn about the \
training data. Probes are cheap and do not count against your patience budget.

# Reply format
Reply with these tags, then exactly one fenced python block containing the \
whole file:

<action>experiment | probe | stop</action>
<kind>{kinds}</kind>
<hypothesis>One sentence: what you are changing and what you expect to happen.</hypothesis>
<rationale>Two or three sentences on why, citing the log or data where you can.</rationale>

Use <action>stop</action> (no code needed) only when you believe further \
experiments will not produce a real gain, and say why in the rationale.
"""


REFERENCE = """
# The bar
{rows}
These were scored on the same validation split through the same harness. The \
strongest one is the number to beat: a result below it is not yet interesting, \
however much it improves on the incumbent. They are other people's fixed \
configurations, not a ceiling."""


def system_prompt(profile: str, budget_s: float, reference: dict[str, float] | None = None) -> str:
    """`reference` maps a short description of each reference model to its
    validation accuracy. Only validation numbers belong here."""
    ref = ""
    if reference:
        rows = "\n".join(f"- {name}: {score:.4f} validation accuracy"
                         for name, score in sorted(reference.items(), key=lambda kv: kv[1]))
        ref = REFERENCE.format(rows=rows)
    return SYSTEM.format(profile=profile, budget=budget_s, kinds=" | ".join(KINDS), reference=ref)


PROBE_FIRST = (
    "# First step\nYou have not looked at the data yet, so this turn must be a probe, not an "
    "experiment. The profile above only has per-feature summaries. Use the probe to learn "
    "something that will change what you try: for example how separable the classes are and "
    "on which features, where the rare classes sit, or how features interact. Print compact "
    "numbers, not raw rows."
)


def curve_summary(curve: list[dict], points: int = 8) -> str:
    if not curve:
        return "(no log() calls)"
    idx = sorted({round(i * (len(curve) - 1) / (points - 1)) for i in range(points)})
    rows = []
    for i in idx:
        row = curve[i]
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k not in ("step", "t"))
        rows.append(f"  step {row['step']} (t={row['t']:.0f}s): {metrics}")
    return "\n".join(rows)


def _outcome(rec: dict) -> str:
    return {
        "accepted": "ACCEPTED (new incumbent)",
        "rejected_screen": "discarded: first seed did not beat incumbent",
        "rejected_confirm": "discarded: gain within seed noise",
        "failed": "failed to run",
    }[rec["status"]]


def user_prompt(state, cfg, probe_first: bool = False) -> str:
    """The per-iteration turn. Only validation numbers appear here; test scores
    are stored under the `hidden` key of each record and are never read."""
    inc = state.incumbent
    parts = [
        f"# Incumbent (from iteration {inc['iter']})\n"
        f"Validation accuracy: mean {inc['val_mean']:.5f} over seeds, "
        f"per seed {[round(v, 5) for v in inc['val']]}.\n\n"
        f"```python\n{state.incumbent_code.strip()}\n```"
    ]
    if state.experiments:
        lines = ["# Experiment log", "iter | kind | outcome | val first seed | val mean | hypothesis"]
        for r in state.experiments:
            first = f"{r['val'][0]:.5f}" if r.get("val") else "-"
            avg = f"{mean(r['val']):.5f}" if len(r.get("val", [])) > 1 else "-"
            lines.append(
                f"{r['iter']} | {r['kind']} | {_outcome(r)} | {first} | {avg} | {r['hypothesis']}"
            )
        parts.append("\n".join(lines))
        last = state.experiments[-1]
        detail = [f"# Detail of the most recent experiment (iteration {last['iter']})",
                  f"Outcome: {_outcome(last)}. {last.get('decision', '')}".strip()]
        if last.get("error"):
            detail.append(f"Error:\n{last['error']}")
        detail.append(f"Logged curve:\n{last.get('curve', '(none)')}")
        if last.get("stdout"):
            detail.append(f"stdout tail:\n{last['stdout'][-1500:]}")
        parts.append("\n".join(detail))
    else:
        parts.append("# Experiment log\nNo experiments yet. The incumbent is the starting baseline.")
    # The first probe shaped the whole run, so it stays visible next to the latest ones.
    for p in {p["iter"]: p for p in state.probes[:1] + state.probes[-2:]}.values():
        parts.append(
            f"# Probe output (iteration {p['iter']}): {p['hypothesis']}\n{p['output'][:4000]}"
        )
    budget = (
        f"# Budget\nThis is iteration {state.iteration + 1} of at most {cfg.max_iters}. "
        f"Experiments since the last accepted improvement: {state.since_improvement} "
        f"(the run stops at {cfg.patience})."
    )
    if cfg.reference:
        gap = max(cfg.reference.values()) - inc["val_mean"]
        budget += (f" The incumbent is {gap:.4f} below the bar." if gap > 0
                   else f" The incumbent is {-gap:.4f} above the bar.")
    parts.append(budget)
    parts.append(PROBE_FIRST if probe_first else "Propose the next step.")
    return "\n\n".join(parts)


def repair_prompt(proposal, code: str, error: str, entry: str) -> str:
    return (
        f"The file you wrote did not run. Fix it and reply with the complete corrected "
        f"file in one fenced python block. Keep the idea being tested the same; only "
        f"repair what is broken. It must still define `{entry}`.\n\n"
        f"Hypothesis under test: {proposal.hypothesis}\n\n"
        f"# Error\n{error}\n\n# File\n```python\n{code.strip()}\n```"
    )


@dataclass
class Proposal:
    action: str  # experiment | probe | stop
    kind: str
    hypothesis: str
    rationale: str
    code: str


class ParseError(ValueError):
    pass


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    if not blocks and text.rstrip().endswith("[reply truncated at max_tokens]"):
        raise ParseError("reply hit the token limit before the code block was complete")
    if not blocks:
        raise ParseError("no fenced python block found in the reply")
    return max(blocks, key=len).strip() + "\n"


def parse_reply(text: str) -> Proposal:
    action = _tag(text, "action").lower()
    hypothesis = " ".join(_tag(text, "hypothesis").split())
    rationale = " ".join(_tag(text, "rationale").split())
    kind = _tag(text, "kind").lower()
    if kind not in KINDS:
        kind = "other"
    if action == "stop":
        return Proposal("stop", kind, hypothesis, rationale, "")
    code = extract_code(text)
    if action not in ("experiment", "probe"):
        action = "probe" if re.search(r"^def probe\(", code, re.M) else "experiment"
    if not hypothesis:
        raise ParseError("missing <hypothesis> tag")
    return Proposal(action, kind, hypothesis, rationale, code)
