"""Wall-clock HP vs PC per model x mode x compression rate (plus average), as a full-width LaTeX table."""

import pandas as pd

MODELS = {"SmolLM2-1.7B": "SmolLM2 1.7B", "OLMo2-1B": "OLMo2 1B"}
MODES = {"distillation": "Distillation", "lm_retraining": "LM retraining"}
COMP_RATES = ["50%", "70%", "90%"]

df = pd.read_csv("wall_clock_times.csv")
t = df.pivot_table(
    index=["model", "mode", "compression_rate"], columns="method", values="wall_clock_h"
)


def cells(hp, pc):
    return f"{hp:.2f} & {pc:.2f} & +{100 * (pc / hp - 1):.0f}\\%"


n = len(COMP_RATES)
comp_heads = [
    rf"\multicolumn{{3}}{{c|}}{{\textbf{{{c.replace('%', r'\%')}}}}}"
    for c in COMP_RATES
]
lines = [
    r"\begin{table*}[t!]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{ll" + "|".join(["ccc"] * (n + 1)) + "}",
    r"\toprule",
    rf"& & \multicolumn{{{3 * n}}}{{c|}}{{\textbf{{Compression Ratio}}}} & \multicolumn{{3}}{{c}}{{}} \\",
    rf"\cmidrule(lr){{3-{2 + 3 * n}}}",
    r"& & " + " & ".join(comp_heads) + r" & \multicolumn{3}{c}{\textbf{Average}} \\",
    " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(n + 1)),
    r"\textbf{Model} & \textbf{Mode} & "
    + " & ".join(["HP & PC & $\\Delta$"] * (n + 1))
    + r" \\",
    r"\midrule",
]
for model, model_label in MODELS.items():
    for i, (mode, mode_label) in enumerate(MODES.items()):
        row = [cells(*t.loc[(model, mode, c), ["HP", "PC"]]) for c in COMP_RATES]
        row.append(cells(*t.loc[(model, mode), ["HP", "PC"]].mean()))
        first = (
            rf"\multirow{{{len(MODES)}}}{{*}}{{\textbf{{{model_label}}}}}"
            if i == 0
            else ""
        )
        lines.append(f"{first} & {mode_label} & " + " & ".join(row) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [
    r"\end{tabular}",
    r"\caption{Wall-clock training time in hours (4$\times$H100, 1B tokens) of hard pruning (HP) "
    r"and projected compression (PC) under different compression ratios. $\Delta$ is the PC overhead relative to HP.}",
    r"\label{tab:wall-clock}",
    r"\end{table*}",
]
print("\n".join(lines))
