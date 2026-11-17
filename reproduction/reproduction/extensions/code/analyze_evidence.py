"""Exact consequences of supplied rate enclosures, with provenance kept explicit."""
from __future__ import annotations
import csv,json,hashlib,shutil
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=list(csv.DictReader((ROOT/'data/legacy_rate_bounds.csv').open()))
    index={(r['model'],F(r['Q']),int(r['live_memory_budget'])):r for r in rows}
    def get(q,m):
        r=index['BB84_real',q,m]
        if r['evidence_status']!='certified_optimum':raise ValueError('Upper not certified')
        return F.from_float(float(r['lower'])),F.from_float(float(r['upper']))
    out=[]
    for Q in sorted({F(r['Q']) for r in rows if r['model']=='BB84_real'}):
        # Every complete real readout with <=5 inputs is feasible by memory 3.
        block_upper=get(Q,3)[1]
        early_upper=5*block_upper
        tail_B=F(1,2**17)
        tail_pump=(1-Q)**160/2
        extra_upper=max(early_upper,tail_B,tail_pump)
        for m in (4,5,6,7):
            U=max(get(Q,m)[1],extra_upper);L=get(Q,m+1)[0]
            if L>U:
                out.append({'Q':str(Q),'low_memory':m,'high_memory':m+1,
                 'fixed_block_upper_used':str(block_upper),'early_stopping_upper':str(early_upper),
                 'long_B_upper':str(tail_B),'long_pump_upper':str(tail_pump),'expanded_low_upper':str(U),'high_lower':str(L),'gap_lower':str(L-U),
                 'ratio_lower':str(L/U),'ratio_lower_decimal':float(L/U),
                 'scope':'Original finite ED/AD benchmark, plus arbitrary public-check-based early stopping and adaptive input order for each fixed real complete readout N<=5, all finite homogeneous four-decoder depths, and all finite constant-decoder raw-pump lengths. Bound is valid for low memory <=7 using the registered terminal reductions. Same original classical coefficient menu. Not a public-hashed-max objective certificate or arbitrary LOCC converse.'})
    targets=[]
    for q,target in [(F(23,100),F(1,2000)),(F(23,100),F(1,1250)),(F(6,25),F(1,100000))]:
        EU=max(5*get(q,3)[1],F(1,2**17),(1-q)**160/2)
        for m in range(4,9):
            lower=get(q,m)[0]
            prev=max(get(q,m-1)[1],EU)
            if prev<target<=lower:
                targets.append({'Q':str(q),'target':str(target),'minimum_memory':m,
                'previous_upper':str(prev),'achievable_lower':str(lower)})
                break
    result={'stopping_robust_separations':out,'certified_target_memory':targets,
       'derivation':'For a fixed N-input measurement, each early-stopped attempt costs D>=1. Completing every stopped attempt with dummy consumed inputs gives an allowed full-block policy with identical useful output reward G and cost N. Therefore max_pi inf_z [G/D]_+ <= N max_pi inf_z [G/N]_+. Taking union N<=5 yields U_early<=5 U_block. Use U_block<=U_legacy,m=3. For homogeneous words at memory<=7, delete terminal Z-decoder levels into classical B rounds; remaining prefix depth<=memory<=7. More than 16 B rounds have rate<=2^-17. At the legal BB84 state z=0, each fresh-pair check accepts with probability<=1-Q, so constant raw-pump N>160 has rate<=(1-Q)^160/2. All arithmetic below uses the exact binary rationals represented by the supplied IEEE-754 outward endpoints.',
       'provenance':'Input endpoints are supplied scientific certificates; independent work here checks exact implications, not a rerun of the original exhaustive optimizer.'}
    (ROOT/'data/stopping_robust_separations.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'n_separations':len(out),'targets':targets},indent=2))
    for r in out:
        if F(r['Q']) in (F(23,100),F(6,25)):print(r)
if __name__=='__main__':main()
