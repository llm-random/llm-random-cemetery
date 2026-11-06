#!/usr/bin/env python
"""Parity audit: markdown tables from parity_numerics.py's JSON results.

    python documents/scripts/parity_tables.py budget FILE.json [FILE.json ...]
    python documents/scripts/parity_tables.py layers layers.json
    python documents/scripts/parity_tables.py final_ce final_ce.json [...]

Budget: one table per (arm, state, split), a row per parameter group, a column
pair (relative error, signed projection) per variant against the fp32 reference,
and the plan's flags: relative error above twice dense's largest group, signed
projection above 1e-3 in magnitude.
"""

import json
import sys

ORDER = [
    "bank_q", "bank_k", "tau", "alpha", "beta", "k_sink", "v_sink", "q_proj",
    "k_proj", "v_proj", "o_proj", "ff", "norm", "embedding", "head",
]


def load_all(paths):
    merged = {}
    for path in paths:
        for key, section in json.load(open(path)).items():
            if key.startswith("_"):
                continue
            target = merged.setdefault(key, {"loss": {}, "vs_ref32": {}, "vs_prod": {}})
            for part in ("loss", "vs_ref32", "vs_prod"):
                target[part].update(section.get(part, {}))
            if "bank_structure" in section:
                target["bank_structure"] = section["bank_structure"]
    return merged


def dense_budget(merged):
    """Twice the largest group relative error of dense_prod against dense_ref32,
    per (state, split)."""
    out = {}
    for key, section in merged.items():
        arm, state, split = key.split("/")
        if arm != "dense" or "dense_prod" not in section["vs_ref32"]:
            continue
        groups = {g: m for g, m in section["vs_ref32"]["dense_prod"].items() if "@" not in g}
        worst = max(groups.items(), key=lambda kv: kv[1]["rel_err"])
        out[(state, split)] = (worst[0], worst[1]["rel_err"])
    return out


def budget(paths):
    merged = load_all(paths)
    budgets = dense_budget(merged)
    for key in sorted(merged):
        section = merged[key]
        arm, state, split = key.split("/")
        variants = list(section["vs_ref32"])
        if not variants:
            continue
        worst = budgets.get((state, split))
        print(f"\n#### {arm}, {state}, {split}\n")
        if worst:
            print(f"Dense's largest group: {worst[0]}, {worst[1]:.2e}; the flag line is {2 * worst[1]:.2e}.\n")
        losses = section["loss"]
        print("Loss (fp64 mean where recorded): " + ", ".join(f"{v} {l:.6f}" for v, l in losses.items()) + "\n")
        print("| group | " + " | ".join(f"{v} rel / proj" for v in variants) + " |")
        print("|---|" + "---|" * len(variants))
        groups = [g for g in ORDER if g in section["vs_ref32"][variants[0]]]
        for g in groups:
            cells = []
            for v in variants:
                m = section["vs_ref32"][v].get(g)
                if m is None:
                    cells.append("")
                    continue
                flag = ""
                if worst and m["rel_err"] > 2 * worst[1]:
                    flag += " **R**"
                if abs(m["proj"]) > 1e-3:
                    flag += " **P**"
                cells.append(f"{m['rel_err']:.2e} / {m['proj']:+.1e}{flag}")
            print(f"| {g} | " + " | ".join(cells) + " |")
        if section.get("vs_prod"):
            print(f"\nAgainst {arm}_prod (relative difference):\n")
            vp = list(section["vs_prod"])
            print("| group | " + " | ".join(vp) + " |")
            print("|---|" + "---|" * len(vp))
            for g in groups:
                cells = [f"{section['vs_prod'][v][g]['rel_err']:.2e} / {section['vs_prod'][v][g]['proj']:+.1e}" if g in section["vs_prod"][v] else "" for v in vp]
                print(f"| {g} | " + " | ".join(cells) + " |")
        if "bank_structure" in section:
            rows = section["bank_structure"]
            keys = ["nonzero_absent_rows", "zero_present_rows", "nonzero_only_target_rows", "max_abs_in_next_absent_rows"]
            agg = {k: max(r[k] for r in rows) for k in keys}
            if "rel_err_present" in rows[0]:
                agg["rel_err_present (max over banks)"] = max(r["rel_err_present"] for r in rows)
                agg["|corr(err, previous row's gradient)| (max)"] = max(abs(r["corr_err_with_prev_row_ref"]) for r in rows)
                for k in ("corr_err_with_next_row_ref", "corr_err_with_own_row_ref", "corr_ref_with_prev_row_ref"):
                    if k in rows[0]:
                        agg[f"|{k}| (max)"] = max(abs(r[k]) for r in rows)
            agg["rows present"] = rows[0]["rows_present"]
            print("\nTier 2 main_grad structure (1.3), worst over the 24 banks: " + json.dumps(agg))


def layers(path):
    data = json.load(open(path))
    print("| layer | prod (compiled) out / dx / dq_bank / dk_bank / dtau / dalpha / dk_sink | std (compiled) out / dx / dq_bank / dtau | ref16 out / dx | kernel flex vs fp32: out / dq / dk / dv / dk_sink / dv_sink | kernel bf16-ref: out / dq | sdpa flash: dq | LSE grad dq | sink weight |")
    print("|---|---|---|---|---|---|---|---|---|")
    for layer, e in sorted(data.items(), key=lambda kv: int(kv[0])):
        p, s, r = e["prod_compiled"], e["std_compiled"], e["ref16"]
        k = e.get("kernel", {})
        f = k.get("flex_vs_ref32", {})
        b = k.get("ref16_vs_ref32", {})
        print(
            f"| {layer} | {p['out']:.1e} / {p['dx']:.1e} / {p.get('dq_bank', float('nan')):.1e} / {p.get('dk_bank', float('nan')):.1e} / {p.get('dtau', float('nan')):.1e} / {p.get('dalpha', float('nan')):.1e} / {p.get('dk_sink', float('nan')):.1e}"
            f" | {s['out']:.1e} / {s['dx']:.1e} / {s.get('dq_bank', float('nan')):.1e} / {s.get('dtau', float('nan')):.1e}"
            f" | {r['out']:.1e} / {r['dx']:.1e}"
            f" | {f.get('out', float('nan')):.1e} / {f.get('dq', float('nan')):.1e} / {f.get('dk', float('nan')):.1e} / {f.get('dv', float('nan')):.1e} / {f.get('k_sink', float('nan')):.1e} / {f.get('v_sink', float('nan')):.1e}"
            f" | {b.get('out', float('nan')):.1e} / {b.get('dq', float('nan')):.1e}"
            f" | {k.get('sdpa_flash_vs_math32', {}).get('dq', float('nan')):.1e}"
            f" | {k.get('lse', {}).get('dq', float('nan')):.1e}"
            f" | {k.get('sink_weight_mean', float('nan')):.4f} |"
        )


def final_ce(paths):
    for path in paths:
        print(json.dumps(json.load(open(path)), indent=1))


if __name__ == "__main__":
    kind, *paths = sys.argv[1:]
    {"budget": budget, "layers": lambda ps: layers(ps[0]), "final_ce": final_ce}[kind](paths)
