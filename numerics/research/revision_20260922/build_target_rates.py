"""Certified attained target-rate inputs, without continuous optimality claims."""
from core import *
from fractions import Fraction as F
from certify_recursive import prepare
from certify_werner_joint_union import terminal
from certify_bb84_recursive import step
from bb84_fixed import physical_kernel
from werner_ablations import exact_sun_output
import stable_intervals as iv
import csv,math


def certify(q,record,kernels,physical):
    ed=record['ED'];ad=record['AD']
    if ed['family']=='Sun_size_zero_flag_ED':
        p,y=exact_sun_output(q,ed);vals=[iv.rational(v) for v in p];scale=iv.rational(y)
    elif ed['family']=='direct_measurement':
        q=F(str(q));vals=[iv.rational(v) for v in (1-F(3,2)*q,q/2,q/2,q/2)];scale=iv.rational(F(1))
    else:
        p,y,_=prepare(F(str(q)),ed)
        vals=[(max(0.,float(iv.down(float(v.lo)))),float(iv.up(float(v.hi)))) for v in p]
        scale=max(0.,float(iv.down(float(y.lo)))),float(iv.up(float(y.hi)))
    pp=tuple(np.array([[v[j] for v in vals]]) for j in (0,1))
    yy=tuple(np.array([v]) for v in scale)
    for _ in range(ad['classical_B_rounds']):
        pp,s=step(pp,pp,'Z');yy=iv.scale(iv.mul(yy,s),F(1,2))
    val=iv.mul(yy,terminal(pp,ad,kernels,physical))
    return dict(lower=float(val[0][0]),upper=float(val[1][0]),ED=ed,AD=ad)


def main():
    kernels={k.entry['id']:k for k in load_codes()};physical={n:physical_kernel(k) for n,k in kernels.items()}
    rec=json.loads((DATA/'werner_v2_full.json').read_text())['results']
    sun=json.loads((DATA/'sun_common_full_AD.json').read_text())['results']
    path=DATA/'attained_target_rates.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results=[],certificates={})
    rows=[]
    for target in (1e-2,1e-3,1e-4,1e-5,1e-6):
        for M in (1,2,4,8):
            candidates=[]
            for src in (rec,sun):
                for v in src:
                    if v['Q']>.3201:continue
                    row=v['results'].get(f'M{M}') or v['results']['M2']
                    if row['rate']>=target:candidates.append((v['Q'],row['rate'],row))
            candidates.sort(key=lambda v:(v[0],v[1]),reverse=True)
            for q,_,row in candidates:
                identity=f'{q:.17g}_'+json.dumps(row['ED'],sort_keys=True)+json.dumps(row['AD'],sort_keys=True)
                if identity not in out['certificates']:
                    out['certificates'][identity]=certify(q,row,kernels,physical)
                    atomic_json(path,out)
                cert=out['certificates'][identity]
                if cert['lower']<target:continue
                rows.append(dict(model='Werner',target=target,M=M,Q=q,lower=cert['lower'],certificate_key=identity,
                                 at_right_edge=q>=.3199));break
    bb=json.loads((DATA/'bb84_policy_curves_real.json').read_text())['results']
    du=json.loads((DATA/'du_Fig15_bb84_recomputed.json').read_text());assert du['complete']
    for target in (1e-2,1e-3,1e-4,1e-5,1e-6):
        for M in (1,2,4,8):
            feasible=[]
            for v in bb.values():
                if v['b']!=v['p']:continue
                control=du['results'][f"{v['b']:.12g}"];assert control['Q']==v['b']
                r=v['results'][f'M{M}'];rate=max(r['lower'],control['lower'])
                if rate>=target:feasible.append((v,rate,control['lower']>r['lower']))
            if not feasible:continue
            v,rate,from_du=max(feasible,key=lambda v:v[0]['b']);r=v['results'][f'M{M}']
            reference=f"du_Fig15_bb84_recomputed.json:{v['b']:.12g}" if from_du else f"{v['b']:.8f}_{v['p']:.8f}:{r.get('policy','direct')}"
            rows.append(dict(model='BB84',target=target,M=M,Q=v['b'],lower=rate,
                             certificate_key=reference,at_right_edge=False))
    out.update(results=rows,BB84_saved_points=len(bb),
               scope='Largest actually certified input among the evaluated construction points, at the stated positive rate. This is an attained lower bound, not the unrestricted or continuous finite-class threshold. Werner search window ends at about Q=.32; right-edge values are marked.')
    atomic_json(path,out)
    with (DATA/'table_attained_target_rates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=r'''\begin{table}[t]
\centering\small
\caption{Largest certified symmetric input among the evaluated construction points that reaches the stated nonzero rate. Entries are rounded downward and are attained lower bounds on a tolerable-error threshold, not upper bounds on any larger class. The BB84 points use a grid of spacing $0.0005$. A dagger marks the edge of the evaluated Werner window near $0.32$. Optional local Pauli degradation extends each guarantee to smaller symmetric errors.}\label{tab:targets}
\begin{tabular}{lccccc}\toprule Input & $R_*$ & $M\leq1$ & $M\leq2$ & $M\leq4$ & $M\leq8$\\\midrule
'''
    for model in ('Werner','BB84'):
        for target in (1e-2,1e-3,1e-4,1e-5,1e-6):
            local={r['M']:r for r in rows if r['model']==model and r['target']==target}
            vals=[]
            for M in (1,2,4,8):
                r=local.get(M)
                vals.append('--' if r is None else f"${math.floor(r['Q']*1e6)/1e6:.6f}"+(r'^{\dagger}' if r['at_right_edge'] else '')+'$')
            text+=model+f' & $10^{{{round(np.log10(target))}}}$ & '+' & '.join(vals)+r'\\'+'\n'
        if model=='Werner':text+=r'\midrule'+'\n'
    text+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    (ROOT/'draft/target_table.tex').write_text(text,encoding='utf-8')
    print('Attained target-rate inputs:',len(rows),'BB84 saved points:',len(bb),flush=True)


if __name__=='__main__':main()
