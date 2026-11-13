"""The next round's sigmas for the AME bank-init sigma grid, by the rules in
documents/notes/ame_bank_sigma_grid.md ("The rounds").

usage:
  python scripts/bank_sigma_next.py ROUND RESULTS_JSON
  python scripts/bank_sigma_next.py ROUND m=g [m=g ...]

ROUND is 2 or 3. RESULTS_JSON is what scripts/bank_sigma_chart.py wrote; its grid
points (not the context runs) are the measured set. Prints the 4 picks in task
order, with the rule that chose each, and the ^sigma_m100 line for
configs/bank_sigma/round<ROUND>.yaml.

Units: m = -100 log10(sigma), so x = log10(sigma) = -m/100. A half-decade is 50
in m, a quarter-decade 25. Lower x is higher m.
"""

import json
import sys

TIE = 0.022
STEP = {2: 50, 3: 25}  # the round's resolution in m


def knees(g, step):
    """Rule 3 of round 2: midpoints of the intervals between measured neighbours,
    largest |delta g| first, on the round's resolution."""
    ms = sorted(g)
    intervals = sorted(zip(ms, ms[1:]), key=lambda ab: -abs(g[ab[1]] - g[ab[0]]))
    for a, b in intervals:
        if (a + b) % 2 == 0 and ((a + b) // 2) % step == 0:
            yield (a + b) // 2, f"knee between m{a} and m{b} (|dg| {abs(g[b] - g[a]):.4f})"


def tie_edges(g, best, step):
    """Rule 3 of round 3: on each side of x*, the point bisecting the gap between
    the outermost tie-band point and the first point outside the band."""
    ms = sorted(g)
    band = g[best] + TIE
    for side, order in (("lower sigma", [m for m in ms if m > best]),
                        ("higher sigma", sorted((m for m in ms if m < best), reverse=True))):
        inside = best
        for m in order:
            if g[m] <= band:
                inside = m
                continue
            if (inside + m) % 2 == 0 and ((inside + m) // 2) % step == 0:
                yield (inside + m) // 2, f"tie edge, {side}: between m{inside} and m{m}"
            break


def next_round(k, g):
    """[(m, rule)] for round k from the measured {m: g}."""
    step = STEP[k]
    best = min(g, key=g.get)
    picks = []

    def add(m, rule):
        if len(picks) < 4 and m > 0 and m not in g and m not in [p[0] for p in picks]:
            picks.append((m, rule))

    if best == max(g):  # x* is the lowest x measured
        add(best + 2 * step, "1: x* is the lowest sigma measured, go lower")
    add(best + step, "2: x* - one step")
    add(best - step, "2: x* + one step")
    if k == 3:
        for m, rule in tie_edges(g, best, step):
            add(m, "3: " + rule)
    for m, rule in knees(g, step):
        add(m, ("4: " if k == 3 else "3: ") + rule)
    # Not in the plan's rules, which can run short: 4 decades have only 3 interior
    # half-decades, two of them x* +- 0.5. Then go one step, then two, beyond the
    # end of the measured range whose g is lower: the side nearer the optimum.
    lo_sigma, hi_sigma = max(g), min(g)
    end, out = (lo_sigma, +step) if g[lo_sigma] <= g[hi_sigma] else (hi_sigma, -step)
    for n in (1, 2):
        add(end + n * out, f"fallback: beyond the better end m{end}")
    return best, picks


def main(argv):
    if len(argv) < 2 or argv[0] not in ("2", "3"):
        sys.exit(__doc__)
    k = int(argv[0])
    if argv[1].endswith(".json"):
        points = json.load(open(argv[1]))["points"]
        g = {p["m"]: p["g"] for p in points if p["m"] is not None and p["g"] is not None}
    else:
        g = {int(a.split("=")[0]): float(a.split("=")[1]) for a in argv[1:]}
    best, picks = next_round(k, g)
    print("measured: " + ", ".join(f"m{m} {g[m]:+.4f}" for m in sorted(g)))
    print(f"x* = m{best} (sigma {10 ** (-best / 100):.3g}); tie band g <= {g[best] + TIE:+.4f}")
    for i, (m, rule) in enumerate(picks):
        print(f"  task {i}: m{m} (sigma {10 ** (-m / 100):.3g})  rule {rule}")
    if len(picks) < 4:
        print(f"  only {len(picks)} picks: the rules ran out")
    print(f"^sigma_m100: [{', '.join(str(m) for m, _ in picks)}]")


if __name__ == "__main__":
    main(sys.argv[1:])
