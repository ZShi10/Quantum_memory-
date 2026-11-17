"""Export published coordinates and exact schedules as LaTeX figure inputs.

Run from any directory with Python 3 and NumPy. No optimizers are run.
Numerical curves are copied from the certified publication tables.
The circuit compiler is replayed only at the two published five-input orders.
"""
from pathlib import Path
from fractions import Fraction
from collections import deque
import ast
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DAT = FIG / "plot_data"
DAT.mkdir(parents=True, exist_ok=True)
SOURCES = set()


def read_table(relative):
    path = ROOT / relative
    SOURCES.add(relative)
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def export(name, columns, rows):
    path = DAT / (name + ".dat")
    with path.open("w", encoding="ascii", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(columns)
        writer.writerows(rows)


def positive(value):
    x = float(value)
    return format(x, ".17g") if x > 0 else "nan"


def line(name, x="Q", y="rate", options=""):
    return (r"\addplot[" + options + r"] table[x=" + x + ",y=" + y +
            r"] {figures/plot_data/" + name + ".dat};\n")


def export_coordinates():
    landscape = read_table("data/global_rate_landscape.csv")
    for model, prefix in [("Werner", "werner"), ("BB84_real", "bb84")]:
        for m in range(1, 9):
            rows = sorted((r for r in landscape if r["source_model"] == model
                           and int(r["memory_budget"]) == m),
                          key=lambda r: Fraction(r["Q"]))
            export(prefix + "_m" + str(m), ["Q", "rate", "gap"],
                   [(format(float(Fraction(r["Q"])), ".17g"), positive(r["lower"]),
                     max(0, float(r["upper"]) - float(r["lower"]))) for r in rows])
        name = prefix + "_linear_Du.csv"
        rows = read_table("reproduction/reproduction/extensions/data/literature/" + name)
        export(prefix + "_du", ["Q", "rate"],
               [(r["Q"], positive(r["rate"])) for r in rows])
    for suffix in ("original", "author"):
        rows = read_table("reproduction/reproduction/extensions/data/literature/"
                          "werner_linear_Sun_" + suffix + ".csv")
        export("werner_sun_" + suffix, ["Q", "rate"],
               [(r["Q"], positive(r["rate"])) for r in rows])
    rows = read_table("data/same_measurement_interval.csv")
    export("same_measurement", ["Q", "R2", "R3"],
           [(r["Q"], r["R2"], r["R3"]) for r in rows])
    rows = read_table("data/revision_20261022_bb84_dense_curves.csv")
    columns = ["Q"] + ["full_R" + str(m) for m in range(2, 9)] + ["real_xyz_R2", "real_xz_R2"]
    export("full_bb84", columns, [[r["Q"]] + [positive(r[c]) for c in columns[1:]] for r in rows])
    path = ROOT / "data/revision_20261022_bb84_bounds.json"
    SOURCES.add("data/revision_20261022_bb84_bounds.json")
    bounds = json.loads(path.read_text())
    lookup = {(r["label"], float(r["Q"]), r["memory"]): r for r in bounds}
    for m in (2, 3, 4, 6):
        rr = [lookup[("full", q, m)] for q in (.17, .19, .2, .21, .22, .23)]
        export("full_upper_m" + str(m), ["Q", "rate", "gap"],
               [(r["Q"], r["lower"], float(r["union_upper"]) - float(r["lower"])) for r in rr])
    for label in ("full", "real_xyz", "real_xz"):
        rr = [lookup[(label, q, 2)] for q in (.17, .2, .23)]
        export(label + "_operation_bounds", ["Q", "rate", "gap"],
               [(r["Q"], r["lower"], float(r["upper"]) - float(r["lower"])) for r in rr])
    return lookup


def make_plots(lookup):
    text = r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=11mm},
qkdaxis,width=.46\textwidth,height=5.8cm,ymode=log,ymin=1e-7,ymax=1.3,
xmin=0,xtick distance=.05,xlabel={Raw error $Q$}]
\nextgroupplot[title={(a) Werner source},xmax=.3334,
ylabel={Net secret bits per raw pair},legend columns=5,
legend style={at={(1.08,-.28)},anchor=north}]
"""
    for m in range(1, 9):
        text += line("werner_m" + str(m), options="mem" + str(m) +
                     ",mark=*,mark size=.65pt,error bars/.cd,y dir=plus,y explicit",
                     y="rate").replace("y=rate]", "y=rate,y error=gap]")
        text += r"\addlegendentry{$m=" + str(m) + "$}\n"
    text += line("werner_du", options="black,dashed") + r"\addlegendentry{Du et al.}" + "\n"
    text += line("werner_sun_original", options="violet,densely dotted") + r"\addlegendentry{Sun et al.}" + "\n"
    text += line("werner_sun_author", options="violet,only marks,mark=o,mark size=1.3pt,forget plot")
    text += r"\nextgroupplot[title={(b) BB84-compatible source},xmax=.25]" + "\n"
    for m in range(8, 0, -1):
        text += line("bb84_m" + str(m), options="mem" + str(m) +
                     ",mark=*,mark size=.65pt,error bars/.cd,y dir=plus,y explicit").replace(
                         "y=rate]", "y=rate,y error=gap]")
    text += line("bb84_du", options="black,dashed")
    text += "\\end{groupplot}\n\\end{tikzpicture}\n\\vspace{25pt}\n"
    (FIG / "landscape.tex").write_text(text, encoding="utf-8")

    text = r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=18mm},
qkdaxis,width=.44\textwidth,height=5.6cm]
\nextgroupplot[title={(a) Common $N\le8$ family},xmin=1.5,xmax=4.5,
ymin=0,ymax=3.1,xtick={2,3,4},xlabel={Local memory $m$},
ylabel={$R$ ($10^{-4}$ bits per raw pair)},ybar,bar width=15pt,bar shift=0pt]
\addplot[fill=mem2,draw=mem2] coordinates {(2,1.7003871645935)};
\addplot[fill=mem3,draw=mem3] coordinates {(3,2.3926642569779)};
\addplot[fill=mem4,draw=mem4] coordinates {(4,2.490684794028)};
\node[anchor=north west,font=\scriptsize] at (rel axis cs:.03,.98)
 {$Q=0.28,\quad R_4/U_3>1.0409$};
\nextgroupplot[title={(b) Same five-input measurement},xmin=.15,xmax=.20,
ymin=.007,ymax=.036,xtick distance=.01,xlabel={Source error $Q$},
ylabel={$R$ (bits per raw pair)},legend pos=north east]
"""
    for m in (2, 3):
        text += line("same_measurement", y="R" + str(m), options="mem" + str(m))
        text += r"\addlegendentry{$m=" + str(m) + "$}\n"
    text += r"""\node[anchor=south west,font=\scriptsize] at (rel axis cs:.03,.03)
 {$R_3/R_2>1.037$ throughout};
\end{groupplot}
\end{tikzpicture}
"""
    (FIG / "matched.tex").write_text(text, encoding="utf-8")

    text = r"""\begin{tikzpicture}
\begin{axis}[qkdaxis,name=top,width=.94\textwidth,height=5.3cm,
title={(a) Full intermediate Clifford operations},
xmin=.16,xmax=.24,ymode=log,ymin=1e-7,ymax=.06,xtick distance=.01,
xlabel={Raw bit and phase error $Q$},ylabel={Secret bits per raw pair},
legend columns=4,legend style={at={(.99,.99)},anchor=north east}]
"""
    for m in range(2, 9):
        text += line("full_bb84", y="full_R" + str(m), options="mem" + str(m))
        text += r"\addlegendentry{$m=" + str(m) + "$}\n"
    text += r"\addlegendimage{only marks,mark=triangle,mark options={rotate=180,fill=white},black}" + "\n"
    text += r"\addlegendentry{Family upper}" + "\n"
    for m in (2, 3, 4, 6):
        text += line("full_upper_m" + str(m), options="only marks,mark=none,mem" + str(m) +
                     ",forget plot,error bars/.cd,y dir=plus,y explicit").replace("y=rate]", "y=rate,y error=gap]")
        # The upper endpoint is drawn explicitly from lower + gap.
        text += (r"\addplot[only marks,mark=triangle,mark options={rotate=180,fill=white},"
                 "mem" + str(m) + r",mark size=1.5pt,forget plot] table[x=Q,y expr=\thisrow{rate}+\thisrow{gap}]"
                 " {figures/plot_data/full_upper_m" + str(m) + ".dat};\n")
    text += r"""\end{axis}
\end{tikzpicture}
\par\vspace{6pt}
\begin{tikzpicture}
\begin{axis}[qkdaxis,width=.44\textwidth,height=4.9cm,
title={(b) Gate and readout control, $m=2$},xmin=.16,xmax=.24,
ymode=log,ymin=1e-6,ymax=.06,xtick={.16,.18,.20,.22,.24},
xlabel={Raw error $Q$},ylabel={Secret bits per raw pair},
legend style={at={(.02,.04)},anchor=south west}]
"""
    for label, style, legend in [("full", "mem2", "Full / XYZ"),
                                  ("real_xyz", "brown,dashed", "Real / XYZ"),
                                  ("real_xz", "black,densely dotted", "Real / XZ")]:
        text += line("full_bb84", y=label + "_R2", options=style)
        text += r"\addlegendentry{" + legend + "}\n"
        text += line(label + "_operation_bounds", options=style +
                     ",only marks,mark=*,mark size=1.2pt,forget plot,error bars/.cd,y dir=plus,y explicit").replace(
                         "y=rate]", "y=rate,y error=gap]")
    text += r"""\end{axis}
\end{tikzpicture}\hspace{6mm}
\begin{tikzpicture}
\begin{axis}[qkdaxis,width=.44\textwidth,height=4.9cm,
title={(c) Guaranteed storage gains},xbar,bar width=7pt,bar shift=0pt,
xmin=0,xmax=48,ymin=-.5,ymax=5.5,y dir=reverse,ytick={0,1,2,3,4,5},
yticklabels={$0.19:\ 2\rightarrow3$,$0.20:\ 2\rightarrow3$,$0.21:\ 2\rightarrow3$,
$0.22:\ 2\rightarrow3$,$0.23:\ 3\rightarrow4$,$0.23:\ 5\rightarrow6$},
ylabel={$Q$: memory change},xlabel={Rate gain (\%)}]
"""
    pairs = [(.19, 2, 3), (.2, 2, 3), (.21, 2, 3), (.22, 2, 3), (.23, 3, 4), (.23, 5, 6)]
    for j, (q, low, high) in enumerate(pairs):
        gain = 100 * (lookup[("full", q, high)]["lower"] / lookup[("full", q, low)]["union_upper"] - 1)
        text += r"\addplot[fill=mem" + str(high) + ",draw=none] coordinates {(" + str(gain) + "," + str(j) + ")};\n"
        shown = int(gain * 100) / 100
        text += r"\node[anchor=west,font=\scriptsize] at (axis cs:" + str(gain + .4) + "," + str(j) + ") {$>" + f"{shown:.2f}" + r"\%$};" + "\n"
    text += "\\end{axis}\n\\end{tikzpicture}\n"
    (FIG / "full_bb84.tex").write_text(text, encoding="utf-8")


def schedule_tex(record, title):
    """Each event remains in its original chronological column."""
    m = record["peak_memory"]
    rows = [[r"\lstick{$q_" + str(i + 1) + r"$}\setwiretype{n}"] for i in range(m)]
    active = [False] * m
    outcome = 0
    for event in record["events"]:
        cells = [r"\setwiretype{" + ("q" if state else "n") + "}" for state in active]
        kind = event["action"]
        if kind == "load":
            slot = event["slot"]
            assert not active[slot]
            cells[slot] = r"\gate{\rho_" + str(event["input"]) + r"}\setwiretype{n}"
            active[slot] = True
        elif kind == "gate":
            slots = event["slots"]
            assert all(active[i] for i in slots)
            if event["gate"][0] == "h":
                cells[slots[0]] = r"\gate{H}\setwiretype{q}"
            else:
                control, target = slots
                cells[control] = r"\ctrl{" + str(target - control) + r"}\setwiretype{q}"
                cells[target] = r"\targ{}\setwiretype{q}"
        elif kind == "measure_Z":
            slot = event["slot"]
            assert active[slot]
            outcome += 1
            cells[slot] = r"\meter{\scriptstyle Z:y_" + str(outcome) + r"}\setwiretype{q}"
            active[slot] = False
        else:
            raise ValueError(kind)
        for row, cell in zip(rows, cells):
            row.append(cell)
    assert not any(active)
    for row in rows:
        row.append(r"\setwiretype{n}")
    return (r"\textbf{" + title + "}\n\n"
            r"\begin{adjustbox}{max width=\linewidth}" + "\n"
            r"\begin{quantikz}[row sep=.22cm,column sep=.055cm,font=\footnotesize]" + "\n" +
            (" \\\\\n".join(" & ".join(row) for row in rows)) +
            "\n\\end{quantikz}\n\\end{adjustbox}\n\\par\\vspace{9pt}\n")


def make_schedules():
    path = ROOT / "reproduction/reproduction/controls/data/executable_control_winners.json"
    SOURCES.add(str(path.relative_to(ROOT)).replace("\\", "/"))
    six = json.loads(path.read_text())
    source = ROOT / "reproduction/reproduction/controls/code/compile_winner.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {"multiply", "gate", "compile_order", "pauli_matrix", "dense_check"}
    module = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef)
                             and n.name in wanted], type_ignores=[])
    scope = {"np": np}
    exec(compile(module, str(source), "exec"), scope)
    text = ""
    checks = []
    for r in six:
        adjacency = [0] * 6
        for a, b in r["edges"]:
            adjacency[a] |= 1 << b
            adjacency[b] |= 1 << a
        residual = float(scope["dense_check"](adjacency, r))
        assert residual < 1e-12
        checks.append({"N": 6, "m": r["peak_memory"], "dense_residual": residual})
        text += schedule_tex(r, "$m=" + str(r["peak_memory"]) + "$")
    (FIG / "werner_schedules.tex").write_text(text, encoding="utf-8")
    adjacency = [18, 5, 10, 4, 1]
    text = ""
    for m, order in [(2, [3, 2, 1, 4, 0]), (3, [3, 2, 4, 0, 1])]:
        r = scope["compile_order"](adjacency, order)
        r["logical_dual"] = 5
        residual = float(scope["dense_check"](adjacency, r))
        assert r["peak_memory"] == m and residual < 1e-12
        checks.append({"N": 5, "m": m, "dense_residual": residual})
        text += schedule_tex(r, "$m=" + str(m) + "$")
    (FIG / "same_measurement_schedules.tex").write_text(text, encoding="utf-8")
    return checks


def decoder_circuits():
    """Find gate words for the three specified canonical logical frames."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    Y = 1j * X @ Z
    H = (X + Z) / np.sqrt(2)
    S = np.diag([1, 1j])
    single = {"I": I, "X": X, "Y": Y, "Z": Z}
    paulis = {a + b: np.kron(single[a], single[b]) for a in single for b in single}
    names = list(paulis)
    mats = np.array(list(paulis.values()))
    c01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex)
    c10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], complex)
    gates = {"H0": np.kron(H,I), "H1": np.kron(I,H), "S0": np.kron(S,I),
             "S1": np.kron(I,S), "C01": c01, "C10": c10}
    def observable(u, label):
        op = u.conj().T @ paulis[label] @ u
        scores = np.abs(np.einsum("aij,ji->a", mats, op)) / 4
        idx = int(np.argmax(scores))
        assert scores[idx] > 1 - 1e-12
        return names[idx]
    def signature(u):
        return tuple(observable(u, p) for p in ("XI", "ZI", "IX", "IZ"))
    queue = deque([(np.eye(4), [])])
    seen = {signature(np.eye(4))}
    targets = {"D_{YY}": ("XX", "YI", "YY"),
               "D_{YZ}": ("XX", "YI", "YZ"),
               "D_{ZY}": ("XX", "ZI", "ZY")}
    found = {}
    while queue and len(found) < 3:
        u, word = queue.popleft()
        triple = tuple(observable(u, p) for p in ("XI", "ZI", "IZ"))
        for name, desired in targets.items():
            if triple == desired and name not in found:
                found[name] = word
        for gate, matrix in gates.items():
            new = matrix @ u
            key = signature(new)
            if key not in seen:
                seen.add(key)
                queue.append((new, word + [gate]))
    assert len(found) == 3
    words = {"D_*": ["S0", "H0", "C01", "S1", "H1"], **found,
             "F": ["H0", "C01"], "G": ["C01", "H0", "C01"]}
    panels = []
    for name, word in words.items():
        rows = [[r"\lstick{0}"], [r"\lstick{1}"]]
        for g in word:
            cells = ["", ""]
            if g.startswith("C"):
                c, t = int(g[1]), int(g[2])
                cells[c] = r"\ctrl{" + str(t - c) + "}"
                cells[t] = r"\targ{}"
            else:
                cells[int(g[1])] = r"\gate{" + g[0] + "}"
            for row, cell in zip(rows, cells):
                row.append(cell)
        rows[0] += ["", r"\rstick{retain}"]
        rows[1] += [r"\meter{Z}", r"\setwiretype{c}"]
        panel = (r"\begin{minipage}{.48\linewidth}\centering" + "\n" +
                 "$" + name + "$\\par\\vspace{3pt}\n" +
                 r"\begin{quantikz}[row sep=.2cm,column sep=.16cm,font=\small]" + "\n" +
                 " \\\\\n".join(" & ".join(row) for row in rows) +
                 "\n\\end{quantikz}\n\\end{minipage}")
        panels.append(panel)
    text = "\n\\par\\vspace{12pt}\n".join(
        panels[i] + "\\hfill\n" + panels[i+1] for i in range(0, len(panels), 2))
    (FIG / "bb84_decoders.tex").write_text(text + "\n", encoding="utf-8")
    return words


if __name__ == "__main__":
    lookup = export_coordinates()
    make_plots(lookup)
    checks = make_schedules()
    words = decoder_circuits()
    report = {"status": "passed", "scope": "figure export and fixed-circuit verification",
              "schedule_checks": checks, "decoder_gate_words": words,
              "source_hashes": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                for p in sorted(SOURCES)}}
    (ROOT / "validation/publication_figures.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "schedule_checks": checks, "decoder_gate_words": words}, indent=2))
