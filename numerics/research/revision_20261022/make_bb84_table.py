"""Round BB84 enclosure endpoints outward for the manuscript table."""
from pathlib import Path
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING
import json
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]

def number(x,upper=False):
    if x is None:return r'--'
    if x==0:return '0'
    d=Decimal.from_float(float(x));exponent=d.adjusted();quantum=Decimal(10)**(exponent-6)
    value=d.quantize(quantum,rounding=ROUND_CEILING if upper else ROUND_FLOOR)
    if exponent>=-2:return format(value,'f')
    mantissa=value/(Decimal(10)**exponent)
    return format(mantissa,'f')+r'\!\times\!10^{'+str(exponent)+'}'

def main():
    data=json.loads((HERE/'export/bb84_bounds.json').read_bytes());lookup={(r['label'],float(r['Q']),r['memory']):r for r in data}
    wanted=[('full',.17,2)]+[('full',q,m) for q in (.19,.2,.21,.22) for m in (2,3)]+[('full',.23,m) for m in (2,3,4,6)]
    wanted += [(label,q,8) for label in ('real_xyz','real_xz') for q in (.17,.2,.23)]
    out=[r'\begin{tabular}{lcccrrr}',r'\toprule',r'Nodes/readout & $Q$ & $m$ & $N$ & $L$ & $U_{\rm bin}$ & $U_{\rm union}$\\',r'\midrule']
    previous=None
    for key in wanted:
        r=lookup.get(key)
        if not r:continue
        label,q,m=key
        if previous is not None and previous!=label:out.append(r'\midrule')
        previous=label;name={'full':'Full/XYZ','real_xyz':'Real/XYZ','real_xz':'Real/XZ'}[label]
        out.append(f'{name} & {q:.2f} & {m} & {r["N"]} & ${number(r["lower"])}$ & ${number(r["upper"],True)}$ & ${number(r["union_upper"],True)}$'+r'\\')
    out += [r'\bottomrule',r'\end{tabular}']
    path=PROJECT/'revised/generated/bb84_clifford_table.tex';path.parent.mkdir(exist_ok=True)
    path.write_text('\n'.join(out)+'\n',encoding='utf-8')
    print(path)
if __name__=='__main__':main()
