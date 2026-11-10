"""Command line: python -m harness {prepare,baselines,run,report}"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="download a dataset and write the fixed splits")
    s.add_argument("--dataset", choices=["covertype", "synthetic"], default="covertype")
    s.add_argument("--data-dir", type=Path)

    s = sub.add_parser("baselines", help="score the reference models through the same runner")
    s.add_argument("--data-dir", type=Path, default=ROOT / "data" / "covertype")
    s.add_argument("--out", type=Path, default=ROOT / "runs" / "baselines.json")
    s.add_argument("--budget", type=float, default=60.0)

    s = sub.add_parser("run", help="start or resume a research run")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--data-dir", type=Path, default=ROOT / "data" / "covertype")
    s.add_argument("--seed-solution", type=Path, default=ROOT / "baselines" / "seed_mlp.py")
    s.add_argument("--backend", choices=["lmstudio", "anthropic"], default="lmstudio")
    s.add_argument("--model")
    s.add_argument("--base-url")
    s.add_argument("--max-iters", type=int, default=30)
    s.add_argument("--patience", type=int, default=8)
    s.add_argument("--budget", type=float, default=60.0, help="training seconds per run")
    s.add_argument("--max-hours", type=float, default=8.0)

    s = sub.add_parser("report", help="write report.md and progress.png for a run")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--baselines", type=Path, default=ROOT / "runs" / "baselines.json")

    a = p.parse_args(argv)

    if a.cmd == "prepare":
        from . import data

        out = a.data_dir or ROOT / "data" / a.dataset
        getattr(data, f"prepare_{a.dataset}")(out)
        print(data.profile(out))

    elif a.cmd == "baselines":
        from statistics import mean

        from . import sandbox, scoring
        from .data import load_meta

        meta, results = load_meta(a.data_dir), {}
        for path in sorted((ROOT / "baselines").glob("*.py")):
            val, test, secs = [], [], []
            for seed in (0, 1, 2):
                res = sandbox.run_fit(
                    path.read_text(encoding="utf-8-sig"), data_dir=a.data_dir,
                    workdir=a.out.parent / "baseline_work" / f"{path.stem}_s{seed}",
                    seed=seed, budget_s=a.budget, meta=meta, grace_s=900,
                )
                if not res.ok:
                    sys.exit(f"{path.name} failed: {res.error}")
                sc = scoring.score(res.preds, a.data_dir)
                val.append(sc["val_acc"]); test.append(sc["test_acc"]); secs.append(res.seconds)
            results[path.stem] = {"val": val, "test": test, "seconds": mean(secs)}
            print(f"{path.stem}: val {mean(val):.5f}  test {mean(test):.5f}  ({mean(secs):.0f}s)")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=2))

    elif a.cmd == "run":
        from . import report
        from .agent import Config, Researcher
        from .llm import make_llm

        cfg = Config(data_dir=a.data_dir, run_dir=a.run_dir, seed_solution=a.seed_solution,
                     max_iters=a.max_iters, patience=a.patience, budget_s=a.budget,
                     max_hours=a.max_hours)
        llm = make_llm(a.backend, a.model, a.base_url)
        print(f"Researcher: {llm.name}  |  run dir: {a.run_dir}")
        Researcher(cfg, llm, echo=lambda m: print(m, flush=True)).run()
        print(f"Report: {report.build(a.run_dir, ROOT / 'runs' / 'baselines.json')}")

    elif a.cmd == "report":
        from . import report

        print(report.build(a.run_dir, a.baselines))


if __name__ == "__main__":
    main()
