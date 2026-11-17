from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'joint_optimization'
EXT=ROOT.parent/'resource_extension'
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
sys.path.insert(1,str(OLD))
