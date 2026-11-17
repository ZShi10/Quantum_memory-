"""Run an unchanged driver with JSON support for NumPy scalar output fields."""
from pathlib import Path
import json,runpy,sys
import numpy as np

default=json.JSONEncoder.default
def encode(self,value):
    if isinstance(value,np.generic):return value.item()
    return default(self,value)
json.JSONEncoder.default=encode
driver=Path(__file__).parent/sys.argv[1]
sys.argv=[str(driver)]+sys.argv[2:]
runpy.run_path(str(driver),run_name='__main__')
