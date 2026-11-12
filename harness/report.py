"""Turn a run's journal into report.md and progress.png.

This is the only place the hidden test scores are read.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from .journal import Journal

# Chart colours: slots 1 and 2 of a colour-blind-validated categorical palette,
# with neutral ink for everything that is not a series.
VAL, TEST, MUTED, GRID, INK, SURFACE = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#0b0b0b", "#fcfcfb"


def _steps(records: list[dict]) -> list[dict]:
    """Accepted records in order: the path the incumbent took."""
    return [r for r in records
            if r.get("type") == "baseline"
            or (r.get("type") == "experiment" and r["status"] == "accepted")]


def _references(baselines_file: Path | None) -> dict:
    if not baselines_file or not baselines_file.exists():
        return {}
    return {k: v for k, v in json.loads(baselines_file.read_text()).items()
            if not k.startswith("_")}


def plot(records: list[dict], out: Path, title: str, bar: tuple[str, float] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = _steps(records)
    exps = [r for r in records if r.get("type") == "experiment" and r.get("val")]
    last_iter = max([r.get("iter", 0) for r in records] + [1])
    floor = steps[0]["val_mean"]
    top = max([s["val_mean"] for s in steps] + [s["hidden"]["test_mean"] for s in steps]
              + [r["val"][0] for r in exps] + ([bar[1]] if bar else []))
    lo = min(floor, steps[0]["hidden"]["test_mean"])
    pad = max((top - lo) * 0.25, 0.002)
    y0, y1 = lo - pad, top + pad

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    rejected = [r for r in exps if r["status"] != "accepted"]
    shown = [r for r in rejected if r["val"][0] >= y0]
    ax.scatter([r["iter"] for r in shown], [r["val"][0] for r in shown], s=26, color=MUTED,
               alpha=0.75, linewidths=0, label="Discarded experiment (first seed)", zorder=2)
    below = len(rejected) - len(shown)

    if bar:  # the reference the researcher was asked to beat: context, so neutral ink
        # Stops at the last iteration so it never runs under the end labels.
        ax.plot([-0.5, last_iter], [bar[1]] * 2, color=MUTED, linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"{bar[0]} reference, validation: {bar[1]:.4f}", (0, bar[1]), xytext=(2, 4),
                    textcoords="offset points", fontsize=8.5, color="#52514e")

    xs = [s["iter"] for s in steps] + [last_iter]
    val_on_top = steps[-1]["val_mean"] >= steps[-1]["hidden"]["test_mean"]
    for key, colour, label in (("val", VAL, "Incumbent, validation"),
                               ("test", TEST, "Incumbent, held-out test")):
        ys = [s["val_mean"] if key == "val" else s["hidden"]["test_mean"] for s in steps]
        ax.step(xs, ys + [ys[-1]], where="post", color=colour, linewidth=2, label=label, zorder=3)
        ax.scatter(xs[:-1], ys, s=42, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        # The two end labels usually sit close together: push them apart vertically.
        above = (key == "val") == val_on_top
        ax.annotate(f"{ys[-1]:.4f}", (last_iter, ys[-1]), xytext=(8, 3 if above else -3),
                    textcoords="offset points", va="bottom" if above else "top",
                    fontsize=9, color=INK, annotation_clip=False)

    ax.set_ylim(y0, y1)
    ax.set_xlim(-0.5, last_iter + max(1.0, last_iter * 0.07))
    ax.set_xlabel("Iteration", color=INK)
    ax.set_ylabel("Accuracy (mean over seeds)", color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    if below:
        ax.text(0.01, 0.02, f"{below} discarded run(s) scored below the plotted range",
                transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _pm(xs: list[float]) -> str:
    return f"{mean(xs):.4f} ± {pstdev(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"


def build(run_dir: Path, baselines_file: Path | None = None) -> Path:
    records = Journal(run_dir).records()
    steps = _steps(records)
    if not steps:
        raise SystemExit(f"no baseline recorded in {run_dir}")
    exps = [r for r in records if r.get("type") == "experiment"]
    probes = [r for r in records if r.get("type") == "probe"]
    stops = [r for r in records if r.get("type") == "stop"]
    first, last = steps[0], steps[-1]
    status = Counter(r["status"] for r in exps)
    repaired = sum(1 for r in exps if r.get("repairs") and r["status"] != "failed")

    refs = _references(baselines_file)
    others = {k: mean(v["val"]) for k, v in refs.items() if mean(v["val"]) > first["val_mean"]}
    best = max(others, key=others.get) if others else None
    plot(records, run_dir / "progress.png", f"Research run: {run_dir.name}",
         (best.replace("_ref", ""), others[best]) if best else None)

    L = [f"# Run report: {run_dir.name}", ""]
    if stops:
        L += [f"Stopped: {stops[-1]['reason']}", ""]
    L += [
        "![progress](progress.png)", "",
        "## Headline", "",
        "| | Validation accuracy | Test accuracy (hidden during the run) |",
        "|---|---|---|",
        f"| Seed solution | {_pm(first['val'])} | {_pm(first['hidden']['test'])} |",
        f"| Final incumbent (iteration {last['iter']}) | {_pm(last['val'])} | {_pm(last['hidden']['test'])} |",
        f"| Gain | {last['val_mean'] - first['val_mean']:+.4f} | "
        f"{last['hidden']['test_mean'] - first['hidden']['test_mean']:+.4f} |",
        "",
        "Mean ± standard deviation over seeds. The researcher model only ever saw the "
        "validation column; the test column measures how much of the gain is real.",
        "",
    ]
    if refs:
        L += ["## Reference models (same splits, same runner)", "",
              "| Model | Validation | Test | Fit time |", "|---|---|---|---|"]
        for name, b in refs.items():
            L.append(f"| {name} | {_pm(b['val'])} | {_pm(b['test'])} | {b['seconds']:.0f}s |")
        L.append("")
    L += [
        "## Activity", "",
        f"- Experiments: {len(exps)} ({status.get('accepted', 0)} accepted, "
        f"{status.get('rejected_screen', 0)} discarded on the first seed, "
        f"{status.get('rejected_confirm', 0)} discarded as within seed noise, "
        f"{status.get('failed', 0)} failed to run)",
        f"- Crashes repaired by the model and then scored: {repaired}",
        f"- Data probes: {len(probes)}",
    ]
    if stops and stops[-1].get("tokens"):
        t = stops[-1]["tokens"]
        L.append(f"- Model tokens in the final session: {t['input']:,} in, {t['output']:,} out")
    L += ["", "## Accepted steps", "",
          "| Iter | Kind | Validation | Test | Hypothesis |", "|---|---|---|---|---|"]
    for s in steps:
        L.append(f"| {s['iter']} | {s['kind']} | {s['val_mean']:.4f} | "
                 f"{s['hidden']['test_mean']:.4f} | {s['hypothesis']} |")
    L += ["", "## Every experiment", "",
          "| Iter | Kind | Outcome | Val (first seed) | Repairs | Hypothesis |", "|---|---|---|---|---|---|"]
    for r in exps:
        v = f"{r['val'][0]:.4f}" if r.get("val") else "-"
        L.append(f"| {r['iter']} | {r['kind']} | {r['status']} | {v} | {r.get('repairs', 0)} | "
                 f"{r['hypothesis']} |")
    out = run_dir / "report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
