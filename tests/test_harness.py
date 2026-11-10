"""Unit tests plus an end-to-end loop driven by a scripted stand-in for the model."""

from pathlib import Path

import pytest

from harness import data, prompts, report, sandbox, scoring
from harness.agent import Config, Researcher
from harness.journal import Journal
from harness.llm import LLMTransientError, Reply

ROOT = Path(__file__).resolve().parent.parent

LOGREG = '''
from sklearn.linear_model import LogisticRegression

def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    model = LogisticRegression(max_iter=200).fit(X_train, y_train)
    log(1, train_acc=model.score(X_train, y_train))
    return model.predict
'''
GBM = '''
from sklearn.ensemble import HistGradientBoostingClassifier

def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    model = HistGradientBoostingClassifier(max_iter=60, random_state=seed).fit(X_train, y_train)
    return model.predict_proba
'''
GBM_BROKEN = GBM.replace("max_iter=60", "max_iter=undefined_name")
CONSTANT = '''
import numpy as np

def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    return lambda X: np.zeros(len(X), dtype=np.int64)
'''
PROBE = '''
import numpy as np

def probe(X_train, y_train):
    print("rows", len(X_train), "classes", np.bincount(y_train).tolist())
'''


def reply(action, code="", hypothesis="h", kind="architecture"):
    body = (f"<action>{action}</action>\n<kind>{kind}</kind>\n"
            f"<hypothesis>{hypothesis}</hypothesis>\n<rationale>because</rationale>\n")
    return body + (f"```python\n{code}\n```" if code else "")


class ScriptedLLM:
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, user):
        self.prompts.append(user)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item, 10, 10)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    data.prepare_synthetic(out)
    return out


def make(tmp_path, dataset, replies, **kw):
    seed = tmp_path / "seed.py"
    seed.write_text(LOGREG)
    cfg = Config(data_dir=dataset, run_dir=tmp_path / "run", seed_solution=seed,
                 budget_s=20, **kw)
    llm = ScriptedLLM(replies)
    return Researcher(cfg, llm, echo=lambda m: None), llm


# -- units ----------------------------------------------------------------------

@pytest.mark.parametrize("snippet", [
    "import os", "import subprocess", "from pathlib import Path", "import socket",
    "x = open('f')", "eval('1')", "import numpy as np\nnp.load('x.npy')",
    "import torch\ntorch.load('m.pt')", "from sklearn.datasets import fetch_covtype",
    "__import__('os')", "def nope(): pass",
])
def test_static_check_rejects(snippet):
    code = snippet + "\ndef fit(*a, **k): pass\n" if "nope" not in snippet else snippet
    assert sandbox.static_check(code)


def test_static_check_accepts_shipped_solutions():
    for path in (ROOT / "baselines").glob("*.py"):
        assert sandbox.static_check(path.read_text()) == [], path.name


def test_parse_reply_variants():
    p = prompts.parse_reply("<think>hmm</think>" + reply("experiment", GBM, "trees win", "architecture"))
    assert (p.action, p.kind, p.hypothesis) == ("experiment", "architecture", "trees win")
    assert "HistGradientBoosting" in p.code
    assert prompts.parse_reply(reply("stop")).action == "stop"
    # No action tag: inferred from the entry point.
    assert prompts.parse_reply(f"<hypothesis>look</hypothesis>\n```python\n{PROBE}\n```").action == "probe"
    with pytest.raises(prompts.ParseError):
        prompts.parse_reply("<action>experiment</action><hypothesis>x</hypothesis> no code")


def test_decide_requires_gain_above_noise():
    assert scoring.decide([0.92, 0.921, 0.919], [0.90, 0.901, 0.899], 0.0005)[0]
    assert not scoring.decide([0.9003, 0.9001, 0.9002], [0.90, 0.90, 0.90], 0.0005)[0]
    # Higher mean, but the candidate's own seeds disagree by more than the gain.
    assert not scoring.decide([0.93, 0.89, 0.90], [0.90, 0.901, 0.899], 0.0005)[0]


def test_timeout_kills_the_run(tmp_path, dataset):
    code = "import time\ndef fit(X_train, y_train, **k):\n    time.sleep(600)\n"
    res = sandbox.run_fit(code, data_dir=dataset, workdir=tmp_path / "w", seed=0,
                          budget_s=2, meta=data.load_meta(dataset), grace_s=6)
    assert res.status == "timeout" and res.seconds < 60


def test_bad_predictions_are_caught(tmp_path, dataset):
    code = "import numpy as np\ndef fit(X_train, y_train, **k):\n    return lambda X: np.zeros(3)\n"
    res = sandbox.run_fit(code, data_dir=dataset, workdir=tmp_path / "w", seed=0,
                          budget_s=20, meta=data.load_meta(dataset))
    assert res.status == "bad_output" and "shape" in res.error


# -- end to end -----------------------------------------------------------------

def test_full_loop(tmp_path, dataset):
    researcher, llm = make(tmp_path, dataset, [
        reply("probe", PROBE, "class balance"),
        LLMTransientError("overloaded"),                 # retried, not fatal
        reply("experiment", GBM_BROKEN, "boosting captures the interaction"),
        f"```python\n{GBM}\n```",                        # the repair
        reply("experiment", CONSTANT, "a constant predictor", "ablation"),
        "I think we should try something else.",         # unusable reply
        reply("stop"),
    ], max_iters=10, patience=5)
    researcher.cfg.llm_retries = 3
    import harness.agent as agent_mod
    agent_mod.time.sleep = lambda s: None  # skip the retry backoff

    reason = researcher.run()
    assert "researcher chose to stop" in reason

    state = Journal(researcher.cfg.run_dir).replay()
    by_iter = {r["iter"]: r for r in state.experiments}
    assert state.probes[0]["output"].startswith("rows")
    assert by_iter[2]["status"] == "accepted" and by_iter[2]["repairs"] == 1
    assert by_iter[3]["status"] == "rejected_screen"
    assert state.incumbent["iter"] == 2 and "HistGradient" in state.incumbent_code
    assert state.incumbent["val_mean"] > 0.85

    # The repair turn carried the traceback; no prompt ever carried a test score.
    assert any("undefined_name" in p and "NameError" in p for p in llm.prompts)
    test_scores = {f"{t:.5f}" for r in Journal(researcher.cfg.run_dir).records()
                   for t in r.get("hidden", {}).get("test", [])}
    val_scores = {f"{v:.5f}" for r in Journal(researcher.cfg.run_dir).records()
                  for v in r.get("val", [])}
    for leaked in test_scores - val_scores:
        assert not any(leaked in p for p in llm.prompts)
    assert "hidden" not in "".join(llm.prompts)

    out = report.build(researcher.cfg.run_dir)
    assert "Final incumbent (iteration 2)" in out.read_text(encoding="utf-8")
    assert (researcher.cfg.run_dir / "progress.png").stat().st_size > 5000


def test_resume_keeps_incumbent(tmp_path, dataset):
    first, _ = make(tmp_path, dataset, [reply("experiment", GBM)], max_iters=1)
    assert "max_iters" in first.run()
    second, llm = make(tmp_path, dataset, [reply("experiment", CONSTANT)], max_iters=2)
    second.run()
    state = Journal(second.cfg.run_dir).replay()
    assert state.incumbent["iter"] == 1 and len(state.experiments) == 2
    assert "HistGradient" in llm.prompts[0]  # resumed prompt shows the accepted code
