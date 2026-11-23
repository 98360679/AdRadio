#!/usr/bin/env python3
"""Emit the paper's untargeted results tables (Tables 5, 6, 7) as CSV from the
embedded plot data, so the Type I/II/III tables are reproducible from the repo.

  docs/table5_typeI.csv    (Experiment 1, Type I)   <- plot_exp1_singlecol.py
  docs/table6_typeII.csv   (Experiment 2, Type II)  <- plot_exp2_sidebyside.py
  docs/table7_typeIII.csv  (Experiment 3, Type III) <- plot_exp3_untargeted.py

Columns: eps, psr_db, then <method>_fool_pct and <method>_ber for each of the six
white-box methods. BER values are exactly as the figures plot them; zero-error
points carry their rule-of-three upper bound (see the plot scripts)."""
import os as _os, sys as _sys
_R = _os.path.abspath(_os.path.dirname(__file__))
while _R != _os.path.dirname(_R) and not _os.path.isdir(_os.path.join(_R, "lib")):
    _R = _os.path.dirname(_R)
_os.chdir(_R)
import os
import csv
import numpy as np

HERE = os.path.join(_R, "plots")


def _load(script):
    ns = {}
    exec(open(os.path.join(HERE, script)).read().split("STYLE =")[0], ns)
    return ns


def _clean(method):
    m = method.lower().replace("$\\alpha$", "a").replace("$", "")
    for a, b in [("(", ""), (")", ""), ("=", ""), (" ", "_")]:
        m = m.replace(a, b)
    return m


def write_table(path, eps, psr, fr, ber, caption):
    methods = list(fr.keys())
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# {caption}"])
        header = ["eps", "psr_db"]
        for m in methods:
            c = _clean(m)
            header += [f"{c}_fool_pct", f"{c}_ber"]
        w.writerow(header)
        for i in range(len(eps)):
            row = [f"{float(eps[i]):g}", f"{float(psr[i]):.2f}"]
            for m in methods:
                row += [fr[m][i], ber[m][i]]
            w.writerow(row)
    print(f"  wrote {path}  ({len(eps)} eps x {len(methods)} methods)")


e1 = _load("plot_exp1_singlecol.py")
e2 = _load("plot_exp2_sidebyside.py")
e3 = _load("plot_exp3_untargeted.py")

# Table 5 — Type I (Experiment 1); PSR = 10 log10(2 eps^2)
E1 = np.asarray(e1["EPS"], float)
write_table("docs/table5_typeI.csv", E1, 10 * np.log10(2 * E1 ** 2),
            e1["UNTARG_FR"], e1["UNTARG_BER"],
            "Table 5: Baseline and Type I Adversary (untargeted)")

# Table 6 — Type II (Experiment 2); PSR column from the data
E2 = np.asarray(e2["EPS"], float)
write_table("docs/table6_typeII.csv", E2, e2["PSR"],
            e2["U_FR"], e2["U_BER"],
            "Table 6: Type II Adversary (untargeted)")

# Table 7 — Type III (Experiment 3); PSR = 20 log10(delta) + 3.01
E3 = np.asarray(e3["DELTA"], float)
D3 = e3["DATA"]
fr3 = {m: D3[m]["fr"] for m in D3}
ber3 = {m: D3[m]["ber"] for m in D3}
write_table("docs/table7_typeIII.csv", E3, 20 * np.log10(E3) + 3.01,
            fr3, ber3, "Table 7: Type III Adversary (untargeted)")
print("done: Tables 5, 6, 7 -> docs/table{5,6,7}_*.csv")
