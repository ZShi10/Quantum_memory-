"""Compile the seven LaTeX figures to standalone PDFs.

Requires TeX with quantikz, PGFPlots, standalone and adjustbox.
Run: python scripts/render_figures.py
Add --png when Poppler's pdftoppm is on PATH.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("protocol", "landscape", "matched", "full_bb84",
         "bb84_decoders", "werner_schedules", "same_measurement_schedules")


def compile_one(name, latex, png):
    build = ROOT / "figures/build"
    output = ROOT / "figures/pdf"
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    preamble = r"""\documentclass[border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,tikz,pgfplots,adjustbox}
\usetikzlibrary{quantikz2,arrows.meta,positioning}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\input{figures/plot_style}
\begin{document}
\begin{minipage}{17.7cm}\centering
"""
    source = preamble + r"\input{figures/" + name + "}\n" + (
        "\\end{minipage}\n\\end{document}\n")
    wrapper = build / (name + "_standalone.tex")
    wrapper.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [latex, "-interaction=nonstopmode", "-halt-on-error",
         "-output-directory=" + str(build), "-jobname=" + name, str(wrapper)],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(name + "\n" + result.stdout[-4000:])
    shutil.copy2(build / (name + ".pdf"), output / (name + ".pdf"))
    if png:
        renderer = shutil.which("pdftoppm")
        if not renderer:
            raise RuntimeError("Put pdftoppm on PATH to use --png.")
        previews = ROOT / "figures/png"
        previews.mkdir(exist_ok=True)
        subprocess.run([renderer, "-singlefile", "-r", "150", "-png",
                        str(output/(name+".pdf")), str(previews/name)], check=True)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    latex = shutil.which("pdflatex")
    if not latex:
        raise SystemExit("Put pdflatex (TeX Live or MiKTeX) on PATH.")
    with ThreadPoolExecutor(max_workers=2) as workers:
        for name in workers.map(lambda n: compile_one(n, latex, args.png), NAMES):
            print("Rendered " + name, flush=True)


if __name__ == "__main__":
    main()
