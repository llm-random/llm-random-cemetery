"""
Adds or updates the generate_latex_table cell at the end of plots_wandb_bgw.ipynb.
Run once, then reload the notebook in Jupyter.
"""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "plots_wandb_bgw.ipynb"

NEW_CELL_SOURCE = '''\
def generate_latex_table(df_eurlex, df_imdb, splits=("0-256", "256-512", "512-inf")):
    """
    Generate LaTeX tables summarising best metrics for EurLex and IMDB.

    EurLex table
    ------------
    Rows    : learning rates (sorted)
    Columns : for each split – Loss (PK-MLM | Baseline) and Micro-F1 (PK-MLM | Baseline)
    Best    : min loss, max micro-F1

    IMDB table
    ----------
    Rows    : learning rates (sorted)
    Columns : for each split – Loss (PK-MLM | Baseline) and Accuracy (PK-MLM | Baseline)
    Best    : min loss, max accuracy

    Parameters
    ----------
    df_eurlex : pd.DataFrame
        Filtered to EurLex runs; needs 'learning_rate', 'model_type',
        'steps/eval/loss_(<split>)', 'steps/eval/micro_f1_(<split>)'.
    df_imdb : pd.DataFrame
        Filtered to IMDB runs; needs 'learning_rate', 'model_type',
        'steps/eval/loss_(<split>)', 'steps/eval/accuracy_(<split>)'.
    splits : tuple[str]
        Token-length split keys to include, e.g. ("0-256", "256-512", "512-inf").

    Returns
    -------
    str
        Two LaTeX \\\\begin{table} … \\\\end{table} blocks, ready to paste.
        Requires \\\\usepackage{booktabs} in the LaTeX preamble.
    """
    import numpy as np

    MODEL_TYPES = ["PK-MLM", "Baseline"]

    def _fmt(v, prec=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.{prec}f}"

    def _fmt_lr(lr):
        if lr is None or (isinstance(lr, float) and np.isnan(lr)):
            return "--"
        return f"{lr:.0e}"

    def _split_label(key):
        return {"0-256": "$<$256", "256-512": "256--512", "512-inf": "$>$512"}.get(key, key)

    def _build_table(df, title, label, metric_label, metric_key_suffix, metric_best_fn):
        lrs = sorted(df["learning_rate"].dropna().unique())
        n = len(splits)
        col_spec = "l" + "rrrr" * n

        lines = []
        lines += [
            f"% ---- {title} ----",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{" + f"{title}: best loss and {metric_label} per learning rate" + r"}",
            r"\label{tab:" + label + r"}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]

        # top header: split spans
        hdr1 = [""] + [r"\multicolumn{4}{c}{" + _split_label(s) + r" tokens}" for s in splits]
        lines.append(" & ".join(hdr1) + r" \\")
        cmidr1 = " ".join(
            r"\cmidrule(lr){" + f"{2 + i*4}-{5 + i*4}" + "}" for i in range(n)
        )
        lines.append(cmidr1)

        # second header: metric spans
        hdr2 = ["LR"] + [f"\\multicolumn{{2}}{{c}}{{Loss}} & \\multicolumn{{2}}{{c}}{{{metric_label}}}"] * n
        lines.append(" & ".join(hdr2) + r" \\")
        cmidr2 = " ".join(
            r"\cmidrule(lr){" + f"{2 + i*4}-{3 + i*4}" + "} "
            r"\cmidrule(lr){" + f"{4 + i*4}-{5 + i*4}" + "}"
            for i in range(n)
        )
        lines.append(cmidr2)

        # model sub-header
        hdr3 = [""] + ["PK-MLM & Baseline & PK-MLM & Baseline"] * n
        lines.append(" & ".join(hdr3) + r" \\")
        lines.append(r"\midrule")

        # data rows
        for lr in lrs:
            row = [_fmt_lr(lr)]
            for split in splits:
                for suffix, fn in [("loss", "min"), (metric_key_suffix, metric_best_fn)]:
                    col = f"steps/eval/{suffix}_({split})"
                    for mt in MODEL_TYPES:
                        if col not in df.columns:
                            row.append("--")
                            continue
                        sub = df[
                            (df["model_type"] == mt) &
                            (df["learning_rate"] == lr) &
                            df[col].notnull()
                        ]
                        val = (sub[col].min() if fn == "min" else sub[col].max()) if not sub.empty else np.nan
                        row.append(_fmt(val))
            lines.append(" & ".join(row) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        return "\n".join(lines)

    eurlex_table = _build_table(df_eurlex, "EurLex", "eurlex_results", "Micro-F1", "micro_f1", "max")
    imdb_table = _build_table(df_imdb, "IMDB", "imdb_results", "Accuracy", "accuracy", "max")

    return eurlex_table + "\n\n" + imdb_table


# ── Usage example ─────────────────────────────────────────────────────────────
# The two DataFrames are built exactly like in the cells above, e.g.:
#
#   micro_f1_metrics = [
#       "steps/eval/loss_(0-256)",   "steps/eval/micro_f1_(0-256)",
#       "steps/eval/loss_(256-512)", "steps/eval/micro_f1_(256-512)",
#       "steps/eval/loss_(512-inf)", "steps/eval/micro_f1_(512-inf)",
#   ]
#   df_eurlex = get_best_values_by_metrics(micro_f1_metrics)
#   df_eurlex = df_eurlex[df_eurlex["name"].str.contains("eurlex")]
#
#   accuracy_metrics = [
#       "steps/eval/loss_(0-256)",   "steps/eval/accuracy_(0-256)",
#       "steps/eval/loss_(256-512)", "steps/eval/accuracy_(256-512)",
#       "steps/eval/loss_(512-inf)", "steps/eval/accuracy_(512-inf)",
#   ]
#   df_imdb = get_best_values_by_metrics(accuracy_metrics)
#   df_imdb = df_imdb[df_imdb["name"].str.contains("imdb")]
#
#   print(generate_latex_table(df_eurlex, df_imdb))
'''

# --------------------------------------------------------------------------- #

with open(NOTEBOOK, "r") as f:
    nb = json.load(f)

print(f"Cells before: {len(nb['cells'])}")

# Filter out any existing cell containing generate_latex_table
nb["cells"] = [c for c in nb["cells"] if "generate_latex_table" not in "".join(c["source"])]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in NEW_CELL_SOURCE.splitlines()],
}
nb["cells"].append(new_cell)

with open(NOTEBOOK, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Cells after:  {len(nb['cells'])}")
print("generate_latex_table cell updated successfully.")
