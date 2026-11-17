from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'joint_optimization'
sys.path.insert(0,str(OLD))
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
