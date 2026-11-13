"""Chart and table of the AME bank-init sigma grid
(documents/notes/ame_bank_sigma_grid.md).

usage:
  python scripts/bank_sigma_chart.py OUT_PREFIX [--ref RUN] [--grid RUN,RUN,...]

Writes OUT_PREFIX.png (the chart) and OUT_PREFIX.json (every number on it), and
prints the doc's Results table.

- Grid runs: every finished run of ideas_cv/nano-ame tagged bank_sigma, unless
  --grid lists run ids. A run's sigma is its logged
  common.attention.ame.bank_init_std; a run older than that key used the code's
  default, 1.0.
- g: the last-500 paired train-loss gap against --ref (default: the finished run
  named bank_sigma_m200, sigma = 0.01 on entropy). Lower is better; within 0.022
  of the best is a tie (AME's same-seed floor).
- Context points, drawn but not used to pick: sigma = 1 on entropy (the three
  parity runs of doi36p8k's config, W&B nano-ame-parity) and the two LEM runs,
  kf4q4t4s (sigma = 0.01) and doi36p8k (sigma = 1).
- Also per run: loss spikes after step 100, the median share of the gradient
  norm that is the banks' (after step 100), and the fraction of steps clipped.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_gap import PROJECT, run_path, series, spikes, window_mean  # noqa: E402

TIE = 0.022
LR = 2.0**-8
PARITY = ["nano-ame-parity/u1zpx7lt", "nano-ame-parity/s65ebe41", "nano-ame-parity/w4hmw00o"]
LEM = ["kf4q4t4s", "doi36p8k"]
GRAD_KEYS = ["steps/train/grad_norm", "steps/train/grad_norm/banks", "steps/train/clip_factor"]

# The dataviz skill's reference light theme, emphasis form: the grid is the one
# series (slot-1 blue); the context points are de-emphasised gray.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"


def fmt_sigma(sigma):
    """0.01, 0.1, 1, 0.00316, 3.16e-4, 1e-5."""
    if sigma >= 1e-3:
        return f"{sigma:.3g}"
    exp = math.floor(math.log10(sigma) + 1e-9)
    return f"{sigma / 10**exp:.3g}e{exp}"


def sigma_of(run):
    ame = (run.config.get("common", {}) or {}).get("attention", {}) or {}
    value = (ame.get("ame", {}) or {}).get("bank_init_std")
    return 1.0 if value is None else float(value)


def readout(api, run_id, ref_loss, n_steps):
    """One run's numbers; g against the reference's loss series."""
    run = api.run(run_path(run_id))
    loss = series(run, "steps/train/loss")
    gap = {s: loss[s] - ref_loss[s] for s in set(loss) & set(ref_loss)}
    g, n_paired = window_mean(gap, n_steps - 500, n_steps)
    rows = list(run.scan_history(keys=["step", *GRAD_KEYS], page_size=10000))
    late = [r for r in rows if int(r["step"]) > 100 and r[GRAD_KEYS[0]]]
    share = float(np.median([r[GRAD_KEYS[1]] / r[GRAD_KEYS[0]] for r in late])) if late else None
    clipped = float(np.mean([r[GRAD_KEYS[2]] < 1 for r in rows])) if rows else None
    name_m = re.fullmatch(r"bank_sigma_m(\d+)", run.name)
    rounds = [t[len("bank_sigma_r"):] for t in run.tags if re.fullmatch(r"bank_sigma_r\d+", t)]
    sigma = sigma_of(run)
    if name_m and not math.isclose(sigma, 10 ** (-int(name_m.group(1)) / 100), rel_tol=1e-9):
        sys.exit(f"{run.name} ({run_id}): logged sigma {sigma} does not match its name")
    return {
        "id": run_id,
        "name": run.name,
        "sigma": sigma,
        "m": int(name_m.group(1)) if name_m else None,
        "round": int(rounds[0]) if rounds else None,
        "g": g,
        "n_paired": n_paired,
        "last_step": max(loss) if loss else None,
        "final_ce": run.summary.get("steps/final_ce"),
        "spikes": len(spikes(loss)),
        "bank_share": share,
        "clipped": clipped,
    }


def find_ref(api):
    runs = [r for r in api.runs(PROJECT, filters={"display_name": "bank_sigma_m200"})
            if r.state == "finished"]
    if len(runs) != 1:
        sys.exit(f"want one finished bank_sigma_m200, found {len(runs)}; pass --ref")
    return runs[0].id


def grid_ids(api):
    return [r.id for r in api.runs(PROJECT, filters={"tags": "bank_sigma"}) if r.state == "finished"]


def style():
    # Inter if this machine has it (matplotlib's cache skips opentype/), else DejaVu.
    for font in sorted(Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf")):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def plot(points, context, best, ref_name, out_png):
    style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.6, 7.2), dpi=200)
    grid = sorted((p for p in points if p["g"] is not None), key=lambda p: p["sigma"])
    xs = [p["sigma"] for p in grid]
    sigmas = xs + [c["sigma"] for c in context]
    lo, hi = min(sigmas) / 2.2, max(sigmas) * 2.2
    panels = [
        ("g", f"Paired train-loss gap vs {ref_name} (σ = 0.01), last 500 steps",
         "gap (nats), lower is better"),
        ("final_ce", "final_ce on the held-out set (bf16: ±0.008 per run)",
         "final_ce (nats), lower is better"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.grid(True, which="major", color=GRIDLINE, lw=0.6)
        ax.set_axisbelow(True)
        if lo < LR < hi:
            ax.axvline(LR, color=AXIS, lw=0.8, zorder=1)
        # Context: sigma = 1 on entropy (mean of three, bar = their range); LEM.
        parity = [c for c in context if c["kind"] == "parity"]
        if parity:
            values = [c[key] for c in parity if c[key] is not None]
            if values:
                ax.vlines(1.0, min(values), max(values), color=MUTED, lw=1.2, zorder=2)
                ax.plot([1.0], [np.mean(values)], ls="none", marker="o", ms=7.5,
                        mfc=SURFACE, mec=MUTED, mew=1.6, zorder=3)
        for c in context:
            if c["kind"] == "lem" and c[key] is not None:
                # Hollow and above the grid's dots, so a LEM point on top of a grid
                # point (kf4q4t4s on bank_sigma_m200) hides neither.
                ax.plot([c["sigma"]], [c[key]], ls="none", marker="D", ms=7.5, mfc="none",
                        mec=MUTED, mew=1.4, zorder=6)
        # The grid.
        ys = [p[key] for p in grid]
        ax.plot(xs, ys, color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round",
                zorder=4)
        ax.plot(xs, ys, ls="none", marker="o", ms=7.5, mfc=BLUE, mec=SURFACE, mew=1.6, zorder=5)
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
    g_ax, ce_ax = axes
    g_ax.axhline(0.0, color=AXIS, lw=0.8, zorder=1)
    if best is not None:
        g_ax.axhspan(best["g"], best["g"] + TIE, color=BLUE, alpha=0.10, lw=0, zorder=1)
        g_ax.annotate(f"best: σ = {fmt_sigma(best['sigma'])}, gap {best['g']:+.4f}",
                      (best["sigma"], best["g"]), xytext=(0, 11), textcoords="offset points",
                      ha="center", va="bottom", color=INK, fontsize=8.5, zorder=7,
                      bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none",
                            "alpha": 0.85})
        g_ax.text(hi / 1.15, best["g"] + TIE, "tie band: within 0.022 of the best",
                  ha="right", va="bottom", color=INK2, fontsize=7.5)
    for ax in axes if lo < LR < hi else []:
        top = ax.get_ylim()[1]
        ax.text(LR * 1.1, top, "σ = LR = 2⁻⁸", ha="left", va="top", color=MUTED, fontsize=7.5)
    ce_ax.set_xscale("log")
    ce_ax.set_xlim(lo, hi)
    decades = [10.0**k for k in range(math.floor(math.log10(lo)) + 1, math.ceil(math.log10(hi)))]
    names = {0: "1", -1: "10⁻¹", -2: "10⁻²", -3: "10⁻³", -4: "10⁻⁴", -5: "10⁻⁵", -6: "10⁻⁶"}
    ce_ax.xaxis.set_major_locator(FixedLocator(decades))
    ce_ax.set_xticklabels([names.get(round(math.log10(d)), f"1e{round(math.log10(d))}")
                           for d in decades])
    ce_ax.xaxis.set_minor_locator(NullLocator())
    ce_ax.set_xlabel("bank init std σ (log scale)")
    handles = [
        Line2D([], [], color=BLUE, lw=2, marker="o", ms=7.5, mfc=BLUE, mec=SURFACE, mew=1.6,
               label="grid run, entropy"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=7.5, mfc=SURFACE, mec=MUTED, mew=1.6,
               label="σ = 1, entropy: mean of 3 (bar: range)"),
        Line2D([], [], color=MUTED, lw=0, marker="D", ms=7.5, mfc="none", mec=MUTED, mew=1.4,
               label="LEM: kf4q4t4s, doi36p8k"),
    ]
    fig.suptitle("AME bank-init σ at h12, 4B tokens", x=0.075, y=0.99, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.075, 0.948, "S2-PS, n = 8 (LR 2⁻⁸), seed 123, one run per σ; "
             "only the bank init differs between runs", ha="left", color=INK2, fontsize=8.5)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.935), ncol=3,
               frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.08, hspace=0.22)
    fig.savefig(out_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_prefix")
    parser.add_argument("--ref", help="reference run id (default: bank_sigma_m200)")
    parser.add_argument("--grid", help="comma list of grid run ids (default: tag bank_sigma)")
    args = parser.parse_args()
    api = wandb.Api(timeout=120)
    ref_id = args.ref or find_ref(api)
    ref = api.run(run_path(ref_id))
    n_steps = int(ref.config["trainer"]["n_steps"])
    ref_loss = series(ref, "steps/train/loss")
    ids = args.grid.split(",") if args.grid else grid_ids(api)
    points = [readout(api, i, ref_loss, n_steps) for i in ids]
    context = [{**readout(api, i, ref_loss, n_steps), "kind": "parity"} for i in PARITY]
    context += [{**readout(api, i, ref_loss, n_steps), "kind": "lem"} for i in LEM]
    measured = [p for p in points if p["g"] is not None]
    best = min(measured, key=lambda p: p["g"]) if measured else None
    for p in points:
        if p["g"] is None:
            p["verdict"] = "no g"
        elif p is best:
            p["verdict"] = "best"
        elif p["g"] <= best["g"] + TIE:
            p["verdict"] = "tie"
        else:
            p["verdict"] = "worse"
    out = Path(args.out_prefix)
    plot(points, context, best, ref.name, out.with_suffix(".png"))
    record = {"ref": {"id": ref_id, "name": ref.name, "n_steps": n_steps}, "tie": TIE,
              "points": points, "context": context}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")

    print("| σ | m | Round | g, last 500 | final_ce | Spikes | Bank share | Clipped | Verdict |")
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for p in sorted(points, key=lambda p: p["sigma"]):
        g = f"{p['g']:+.4f}" if p["g"] is not None else "-"
        ce = f"{p['final_ce']:.4f}" if p["final_ce"] is not None else "-"
        share = f"{p['bank_share']:.2f}" if p["bank_share"] is not None else "-"
        clipped = f"{p['clipped']:.0%}" if p["clipped"] is not None else "-"
        print(f"| {fmt_sigma(p['sigma'])} | {p['m'] or '-'} | {p['round'] or '-'} | {g} | {ce} "
              f"| {p['spikes']} | {share} | {clipped} | {p['verdict']} |")
    print()
    for c in context:
        g = f"{c['g']:+.4f}" if c["g"] is not None else "-"
        print(f"context {c['kind']:6s} {c['id']:28s} {c['name']:32s} σ {fmt_sigma(c['sigma'])} "
              f"g {g} final_ce {c['final_ce']}")


if __name__ == "__main__":
    main()
