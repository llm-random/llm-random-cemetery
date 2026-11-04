"""Dump wall-clock times of the best 1B-token HP/PC runs (OLMo2, SmolLM2, Qwen3) to wall_clock_times.csv."""

import wandb, pandas as pd

api = wandb.Api()
P = "ideas_cv/llm-random-test"
COMP = {"10p": "90%", "30p": "70%", "50p": "50%"}
LOSS = "steps/100/train/loss"


def row(r, model, selection):
    t = set(r.tags)
    s = r.summary._json_dict
    hist = [
        h["steps/100/time"]
        for h in r.scan_history(keys=["steps/100/time"])
        if h.get("steps/100/time") is not None
    ]
    n_steps = r.config.get("trainer", {}).get("n_steps")
    mean_step = sum(hist) / len(hist) if hist else None
    return {
        "model": model,
        "method": "HP" if "hard_pruning" in t else "PC",
        "mode": "distillation" if "distillation" in t else "lm_retraining",
        "compression_rate": next(COMP[c] for c in COMP if c in t),
        "pct_tag": next(c for c in COMP if c in t),
        "wall_clock_s": s.get("_runtime"),
        "wall_clock_h": s.get("_runtime", 0) / 3600,
        "n_steps": n_steps,
        "final_step": s.get("step"),
        "mean_s_per_step": mean_step,
        "train_time_est_s": mean_step * n_steps if mean_step and n_steps else None,
        "final_train_loss": s.get(LOSS),
        "gpu": r.metadata.get("gpu"),
        "gpu_count": r.metadata.get("gpu_count"),
        "host": r.metadata.get("host"),
        "state": r.state,
        "run_id": r.id,
        "run_name": r.name,
        "selection": selection,
        "tags": " ".join(sorted(t)),
    }


rows = []
# OLMo2 / SmolLM2: "best" tag; best_late* reruns supersede plain best, finished beats failed
for fam, model in [("olmo2", "OLMo2-1B"), ("smollm2", "SmolLM2-1.7B")]:
    runs = list(
        api.runs(
            P, filters={"$and": [{"tags": "best"}, {"tags": fam}, {"tags": "1B_train"}]}
        )
    )
    groups = {}
    for r in runs:
        t = set(r.tags)
        key = (
            "hard_pruning" in t,
            "distillation" in t,
            next(c for c in COMP if c in t),
        )
        rank = (
            r.state == "finished",
            max([2 if "best_late2" in t else 1 if "best_late" in t else 0]),
        )
        if key not in groups or rank > groups[key][0]:
            groups[key] = (rank, r)
    for _, r in groups.values():
        rows.append(
            row(
                r,
                model,
                "wandb tag best"
                + (
                    " (best_late)"
                    if any(x.startswith("best_late") for x in r.tags)
                    else ""
                ),
            )
        )

# Qwen: no "best" tag -> lowest final train loss per group (same filters as notebook)
runs = api.runs(
    P, filters={"$and": [{"tags": "qwen"}, {"tags": "1.7B"}, {"tags": "1B_train"}]}
)
groups = {}
for r in runs:
    t = set(r.tags)
    if t & {"remove", "pc_weights_lr_ratio"} or r.state != "finished":
        continue
    loss = r.summary._json_dict.get(LOSS)
    if loss is None:
        continue
    key = ("hard_pruning" in t, "distillation" in t, next(c for c in COMP if c in t))
    if key not in groups or loss < groups[key][0]:
        groups[key] = (loss, r)
for _, r in groups.values():
    rows.append(row(r, "Qwen3-1.7B", "lowest final train loss (no best tag)"))

df = pd.DataFrame(rows).sort_values(["model", "mode", "compression_rate", "method"])
df.to_csv("wall_clock_times.csv", index=False)
print(
    df[
        [
            "model",
            "mode",
            "compression_rate",
            "method",
            "wall_clock_h",
            "mean_s_per_step",
            "gpu",
            "gpu_count",
            "state",
            "run_id",
            "selection",
        ]
    ].to_string()
)
