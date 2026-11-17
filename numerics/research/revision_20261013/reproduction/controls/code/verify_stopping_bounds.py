"""Recheck exact stopping-transfer inequalities from frozen source endpoints.

This replays the new analytic consequence, not the upstream class optimizer.
"""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]

def run():
 points=json.loads((ROOT/'source_provenance/selected_archived_points.json').read_text());rows=[]
 for q,small,large,name,target in [(.23,4,5,'BB23_m5',F(4,10000)),(.23,5,6,'BB23_m6',None),(.24,5,6,'BB24_m6',F(1,100000))]:
  p=points['BB84_real_'+str(q)];r=p['results'][f'M{small}']
  upper=max(F.from_float(r['registered_upper']),5*F.from_float(r['complete_block_upper']))
  cert=json.loads((ROOT/'verification'/f'{name}_certificate.json').read_text());lower=F(cert['target_lower'])
  assert lower>upper
  if target is not None:assert upper<target<=lower
  rows.append({'Q':q,'smaller_memory':small,'larger_memory':large,'exact_stopping_upper':str(upper),'independent_lower_target':str(lower),'ratio_strictly_above':float(lower/upper),'target_rate':str(target) if target else None,'target_minimum_memory':large if target else None})
 report={'status':'all_exact_stopping_and_target_rate_inequalities_pass','scope':'New consequence of archived broad upper bounds and newly generated fixed-protocol lower covers; not a replay of all archived optimizer certificates.','comparisons':rows}
 (ROOT/'verification/stopping_bounds_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 return report
if __name__=='__main__':run()
