"""TeX Live fallback retaining raw bytes for mixed-encoding Windows logs."""
from pathlib import Path
import subprocess,shutil,json
ROOT=Path(__file__).resolve().parent;FIG=ROOT/'figures';OUT=ROOT.parents[1]/'output'/'pdf'
compiler=shutil.which('pdflatex')
if compiler is None:raise RuntimeError('No pdflatex installation found.')
command=[compiler,'-interaction=nonstopmode','-halt-on-error','-synctex=1',f'-output-directory={OUT.as_posix()}','research_numerics.tex']
run=subprocess.run(command,cwd=FIG,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(FIG/'compile_console.log').write_bytes(run.stdout)
print(run.stdout.decode('utf-8',errors='replace')[-2000:])
if run.returncode:raise SystemExit(run.returncode)
log=(OUT/'research_numerics.log').read_text(encoding='utf-8',errors='replace')
assert 'Output written on ' in log
assert 'Overfull' not in log
print(json.dumps(dict(status='passed',pdf=str(OUT/'research_numerics.pdf'))))
