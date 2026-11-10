"""The research loop: propose -> check -> run -> repair -> score -> accept or discard."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from . import prompts, sandbox, scoring
from .data import load_meta, profile
from .journal import Journal
from .llm import LLM, LLMFatalError, LLMTransientError, Reply


@dataclass
class Config:
    data_dir: Path
    run_dir: Path
    seed_solution: Path
    max_iters: int = 30
    patience: int = 8          # experiments without an accepted gain before stopping
    budget_s: float = 60.0     # training wall-clock per run
    seeds: tuple[int, ...] = (0, 1, 2)
    max_repairs: int = 2       # times a crashing file is sent back to be fixed
    min_delta: float = 0.0005  # smallest mean gain worth accepting
    max_hours: float = 8.0
    llm_retries: int = 6


class Researcher:
    def __init__(self, cfg: Config, llm: LLM, echo=print):
        self.cfg, self.llm, self.echo = cfg, llm, echo
        self.journal = Journal(cfg.run_dir)
        self.meta = load_meta(cfg.data_dir)
        self.system = prompts.system_prompt(profile(cfg.data_dir), cfg.budget_s)
        self.tokens = [0, 0]

    # -- model calls ---------------------------------------------------------

    def _ask(self, user: str) -> Reply:
        """Call the researcher model, backing off on transient failures."""
        delay = 5.0
        for attempt in range(1, self.cfg.llm_retries + 1):
            try:
                reply = self.llm.complete(self.system, user)
                self.tokens[0] += reply.input_tokens
                self.tokens[1] += reply.output_tokens
                return reply
            except LLMTransientError as e:
                if attempt == self.cfg.llm_retries:
                    raise LLMFatalError(f"gave up after {attempt} attempts: {e}") from e
                self.echo(f"  model call failed ({e}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise AssertionError("unreachable")

    # -- execution -----------------------------------------------------------

    def _run_seeds(self, code: str, tag: str, seeds):
        """Run `code` on each seed. Returns (val scores, test scores, last result or None)."""
        val, test, res = [], [], None
        for seed in seeds:
            res = sandbox.run_fit(
                code, data_dir=self.cfg.data_dir, workdir=self.cfg.run_dir / "work" / f"{tag}_s{seed}",
                seed=seed, budget_s=self.cfg.budget_s, meta=self.meta,
            )
            if not res.ok:
                return val, test, res
            s = scoring.score(res.preds, self.cfg.data_dir)
            val.append(s["val_acc"])
            test.append(s["test_acc"])
            self.echo(f"  seed {seed}: val {s['val_acc']:.5f} ({res.seconds:.0f}s)")
        return val, test, res

    def _baseline(self) -> None:
        code = self.cfg.seed_solution.read_text(encoding="utf-8-sig")
        self.echo("Baseline: running the seed solution")
        val, test, res = self._run_seeds(code, "iter000", self.cfg.seeds)
        if not res.ok:
            raise RuntimeError(f"seed solution failed: {res.error}")
        self.journal.append({
            "type": "baseline", "iter": 0, "kind": "baseline",
            "hypothesis": "Starting point supplied with the harness.",
            "code_file": self.journal.save_code("iter000", code),
            "status": "accepted", "val": val, "val_mean": mean(val),
            "hidden": {"test": test, "test_mean": mean(test)},
        })

    def _first_run_with_repairs(self, proposal, tag: str, entry: str, run):
        """Run once; on failure hand the error back to the model for a fix."""
        code, repairs = proposal.code, 0
        while True:
            res = run(code, f"{tag}_r{repairs}")
            if res.ok or repairs >= self.cfg.max_repairs:
                return code, res, repairs
            repairs += 1
            self.echo(f"  {res.status}: asking for repair {repairs}/{self.cfg.max_repairs}")
            reply = self._ask(prompts.repair_prompt(proposal, code, res.error, entry))
            try:
                code = prompts.extract_code(reply.text)
            except prompts.ParseError as e:
                res.error = f"repair reply unusable: {e}"
                return code, res, repairs

    def _probe(self, it: int, proposal) -> None:
        code, res, repairs = self._first_run_with_repairs(
            proposal, f"iter{it:03d}", "probe",
            lambda c, tag: sandbox.run_probe(
                c, data_dir=self.cfg.data_dir, workdir=self.cfg.run_dir / "work" / tag),
        )
        output = res.stdout_tail if res.ok else f"Probe failed: {res.error}"
        self.journal.append({
            "type": "probe", "iter": it, "hypothesis": proposal.hypothesis,
            "code_file": self.journal.save_code(f"iter{it:03d}_probe", code),
            "status": res.status, "repairs": repairs, "output": output or "(no output)",
        })

    def _experiment(self, it: int, proposal, state) -> None:
        cfg, inc = self.cfg, state.incumbent
        tag = f"iter{it:03d}"
        seed0 = cfg.seeds[0]
        code, res, repairs = self._first_run_with_repairs(
            proposal, tag, "fit",
            lambda c, t: sandbox.run_fit(
                c, data_dir=cfg.data_dir, workdir=cfg.run_dir / "work" / f"{t}_s{seed0}",
                seed=seed0, budget_s=cfg.budget_s, meta=self.meta),
        )
        rec = {
            "type": "experiment", "iter": it, "kind": proposal.kind,
            "hypothesis": proposal.hypothesis, "rationale": proposal.rationale,
            "code_file": self.journal.save_code(tag, code), "repairs": repairs,
            "incumbent_iter": inc["iter"], "incumbent_val_mean": inc["val_mean"],
            "curve": prompts.curve_summary(res.curve), "stdout": res.stdout_tail[-1500:],
            "seconds": round(res.seconds, 1),
        }
        if not res.ok:
            rec.update(status="failed", error=f"[{res.status}] {res.error}")
            self.echo(f"  failed after {repairs} repair(s): {res.status}")
            self.journal.append(rec)
            return

        s = scoring.score(res.preds, cfg.data_dir)
        val, test = [s["val_acc"]], [s["test_acc"]]
        self.echo(f"  seed {seed0}: val {val[0]:.5f} vs incumbent mean {inc['val_mean']:.5f}")
        if not scoring.screen(val[0], inc["val"]):
            rec.update(status="rejected_screen",
                       decision=f"First seed {val[0]:.5f} <= incumbent mean {inc['val_mean']:.5f}.")
        else:
            more_val, more_test, more = self._run_seeds(code, tag, cfg.seeds[1:])
            val += more_val
            test += more_test
            if more is not None and not more.ok:
                rec.update(status="failed",
                           error=f"[{more.status}] on a confirmation seed: {more.error}")
            else:
                accepted, why = scoring.decide(val, inc["val"], cfg.min_delta)
                rec.update(status="accepted" if accepted else "rejected_confirm", decision=why)
        rec.update(val=val, val_mean=mean(val), hidden={"test": test, "test_mean": mean(test)})
        self.echo(f"  -> {rec['status']}  {rec.get('decision', rec.get('error', ''))}")
        self.journal.append(rec)

    # -- main loop -----------------------------------------------------------

    def run(self) -> str:
        cfg = self.cfg
        state = self.journal.replay()
        if state.incumbent is None:
            self._baseline()
        elif state.iteration:
            self.journal.append({"type": "resume", "iter": state.iteration})
            self.echo(f"Resuming after iteration {state.iteration}")
        deadline = time.time() + cfg.max_hours * 3600
        bad_replies = 0

        while True:
            state = self.journal.replay()
            if state.iteration >= cfg.max_iters:
                reason = f"reached max_iters={cfg.max_iters}"
            elif state.since_improvement >= cfg.patience:
                reason = f"converged: {cfg.patience} experiments without an accepted gain"
            elif time.time() > deadline:
                reason = f"reached the {cfg.max_hours}h wall-clock limit"
            elif bad_replies >= 3:
                reason = "three unusable model replies in a row"
            else:
                reason = None
            if reason:
                break

            it = state.iteration + 1
            self.echo(f"\n[{it}/{cfg.max_iters}] incumbent val {state.incumbent['val_mean']:.5f} "
                      f"| {state.since_improvement}/{cfg.patience} since last gain")
            try:
                reply = self._ask(prompts.user_prompt(state, cfg))
                try:
                    proposal = prompts.parse_reply(reply.text)
                except prompts.ParseError as e:
                    bad_replies += 1
                    self.echo(f"  unusable reply: {e}")
                    self.journal.append({"type": "invalid_reply", "iter": it, "error": str(e),
                                         "reply_head": reply.text[:500]})
                    continue
                bad_replies = 0
                self.echo(f"  {proposal.action}/{proposal.kind}: {proposal.hypothesis}")
                if proposal.action == "stop":
                    reason = f"researcher chose to stop: {proposal.rationale}"
                    break
                if proposal.action == "probe":
                    self._probe(it, proposal)
                else:
                    self._experiment(it, proposal, state)
            except LLMFatalError as e:
                reason = f"model backend unavailable: {e}"
                break
            except Exception as e:  # a harness bug must not end an unattended run
                bad_replies += 1
                self.echo(f"  harness error, skipping iteration: {type(e).__name__}: {e}")
                self.journal.append({"type": "harness_error", "iter": it,
                                     "error": f"{type(e).__name__}: {e}"})

        self.journal.append({"type": "stop", "iter": state.iteration, "reason": reason,
                             "tokens": {"input": self.tokens[0], "output": self.tokens[1]}})
        self.echo(f"\nStopped: {reason}")
        return reason
