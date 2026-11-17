"""Run already-reviewed, independent validation tasks sequentially."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
for job in ['controls-check','theory-checks','pumping-checks','bb84-block-control','extended-endpoints']:
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'run.py'),job],cwd=ROOT,check=True)
