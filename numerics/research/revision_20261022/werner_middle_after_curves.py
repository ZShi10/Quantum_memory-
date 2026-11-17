"""Use a released dense-curve worker for the middle Werner interval."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent
status=HERE/'curves_extended/status.json'
while not(status.exists() and json.loads(status.read_bytes()).get('stage')=='complete'):time.sleep(15)
raise SystemExit(subprocess.call([sys.executable,'-X','utf8',str(HERE/'werner_campaign.py'),
    '--workers','1','--min-q','.3','--max-q','.32','--label','middle']))
