"""Export only completed, traceable controls for the manuscript."""
from pathlib import Path
from fractions import Fraction as F
import csv, hashlib, json, shutil
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
OUT=PROJECT/'revised/data'


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];sources={}
    def record(q,low_m,high_m,U,L,paths):
        assert L>U>=0
        rows.append(dict(Q=str(q),low_memory=low_m,high_memory=high_m,
            upper_low=U,lower_high=L,guaranteed_relative_gain=float(F.from_float(L)/F.from_float(U)-1),
            strict=True))
        for p in paths:sources[str(p.relative_to(PROJECT))]=sha(p)
    block=HERE/'full_bb84/block_upper/0.17/receipt.json'
    reg=HERE/'full_bb84/registered_refined/0.17/receipt.json'
    b,r=read(block),read(reg);assert b['complete'] and r['complete']
    record(.17,2,3,max(b['results']['M2']['upper'],r['results']['M2']['upper']),
           r['results']['M3']['lower'],[block,reg])
    for q in [.175,.18]:
        p=HERE/'full_bb84'/('block_refined' if q==.18 else 'block_separation')/str(q)/'receipt.json'
        x=read(p);assert x['complete'] and x['strict']
        record(q,x['memory']-1,x['memory'],x['union_low_upper'],x['high_lower'],[p])
    p=HERE/'full_bb84/block_refined/0.2/receipt.json'
    reg=HERE/'full_bb84/registered_refined/0.2/receipt.json'
    x,r=read(p),read(reg);assert x['complete'] and r['complete']
    # The adjacent 1-versus-2 gap did not close. A five-slot feasible witness
    # does strictly exceed this SAME complete one-slot upper. No claim that
    # five slots are necessary follows from that nonadjacent comparison.
    record(.2,1,5,x['union_low_upper'],r['results']['M5']['lower'],[p,reg])
    unresolved=dict(Q=.2,low_memory=1,high_memory=2,upper_low=x['union_low_upper'],
                    lower_high=x['high_lower'],strict=x['strict'])
    high=HERE/'full_bb84/block_high_noise_refined/0.23/receipt.json'
    if high.exists():
        x=read(high)
        if x['complete'] and x['strict']:
            record(.23,x['memory']-1,x['memory'],x['union_low_upper'],x['high_lower'],[high])
    summary=dict(complete=True,rows=rows,unresolved_adjacent_control=unresolved,sources=sources,
        family='Full-Clifford complete N<=5 readouts, plus registered four-real-decoder recurrences/trees/pumps with XYZ terminals, full n<=8 and long endpoint AD/OTP menu. Fixed hidden-z policy. Attempt and retry rules matched across budgets.',
        interpretation='Valid whole-lower-memory exclusions at specified Q; no optimal rate or continuous source-Q interval is inferred.')
    (OUT/'full_bb84_controls.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'full_bb84_controls.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    curve=HERE/'full_bb84/fixed_curves/receipt.json'
    extended=HERE/'full_bb84/fixed_curves_extended/receipt.json'
    if extended.exists() and read(extended)['complete']:curve=extended
    if curve.exists() and read(curve)['complete']:
        for name,dest in [('curves.csv','full_bb84_fixed_curves.csv'),('policies.json','full_bb84_fixed_policies.json'),
                          ('receipt.json','full_bb84_fixed_curves_receipt.json')]:
            shutil.copyfile(curve.parent/name,OUT/dest)
    replay=HERE/'wide_werner_derivative_v2/replay.json'
    assert read(replay)['status']=='passed'
    shutil.copyfile(replay,OUT/'matched_werner_wider_interval.json')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
