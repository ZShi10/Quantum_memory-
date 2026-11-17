"""Assemble union certificates and manuscript tables with directed rounding."""
from core import *
import csv,math


def load(name):
    path=DATA/name
    return json.loads(path.read_text()) if path.exists() else None


def lower_tex(x,digits=5):
    if x<=0:return '$0$'
    e=math.floor(math.log10(x));v=math.floor(x/10**e*10**(digits-1))/10**(digits-1)
    return f'${v:.{digits-1}f}\\times10^{{{e}}}$'


def upper_tex(x,digits=2):
    if x<=0:return '$0$'
    e=math.floor(math.log10(x));v=math.ceil(x/10**e*10**(digits-1))/10**(digits-1)
    return f'${v:.{digits-1}f}\\times10^{{{e}}}$'


def main():
    core=load('werner_joint_union_core_certified.json')
    rec=load('werner_v2_full_menu_certified.json')
    union=[]
    for q,r in sorted(rec['results'].items(),key=lambda x:float(x[0])):
        if q not in core['results']:continue
        for M in range(1,9):
            key=f'M{M}';c=core['results'][q]['results'][key];rr=r['results'][key]
            lo=max(c['union_lower'],float(rr['lower']));hi=max(c['core_upper'],float(rr['upper']))
            assert hi+1e-14>=lo
            union.append(dict(model='Werner',permission='X/Y/Z',Q=float(q),M=M,lower=lo,upper=hi,gap=max(0.,hi-lo)))
    for q in (.15,.18,.2,.22,.24):
        r=load(f'bb84_joint_extension_real_{q:.5f}_{q:.5f}_n128_menu_certified.json')
        c=load(f'bb84_block_relaxation_real_{q:.5f}_{q:.5f}_n128.json')
        assert r['complete'] and c['complete']
        for M in range(1,9):
            rr=r['results'][f'M{M}'];hi=max(rr['upper'],c['results'][f'M{min(M,2)}']['upper'])
            lo=rr['lower'];assert hi>=lo
            union.append(dict(model='BB84',permission='X/Z',Q=q,M=M,lower=lo,upper=hi,gap=hi-lo))
    with (DATA/'table_joint_union.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(union[0]));writer.writeheader();writer.writerows(union)
    atomic_json(DATA/'joint_union_summary.json',dict(results=union,
         scope='Union of the complete fixed terminal core and registered recursion/pumping, at saved anchors only. Werner enlarged final-axis permission; BB84 real fixed compound policies.',
         complete_Werner_anchors=len({r['Q'] for r in union if r['model']=='Werner'})))
    text=r'''\begin{table}[t]
\centering\small
\caption{Certified union rates at selected inputs. Entries are feasible lower bounds rounded downward. The last column bounds the largest absolute optimality gap among the three displayed budgets. Full intervals for all eight budgets are supplied as data. Werner permits final $X/Y/Z$ measurements; BB84 fixes real operations and $X/Z$ measurements, with $b=p=Q$. The union and its limits are defined in Sec.~\ref{sec:class}.}\label{tab:rates}
\begin{tabular}{lccccc}
\toprule Input & $Q$ & $M\leq2$ & $M\leq4$ & $M\leq8$ & Max. gap\\\midrule
'''
    for model in ('Werner','BB84'):
        for q in sorted({r['Q'] for r in union if r['model']==model}):
            rows={r['M']:r for r in union if r['model']==model and r['Q']==q}
            text+=f'{model} & {q:.2f} & '+' & '.join(lower_tex(rows[m]['lower']) for m in (2,4,8))+' & '+upper_tex(max(rows[m]['gap'] for m in (2,4,8)))+r'\\'+'\n'
        if model=='Werner':text+=r'\midrule'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\par\smallskip\footnotesize
At Werner $Q=0.32$, the small-$M$ rate is much smaller than its absolute gap; no high relative-precision optimum is claimed there. A zero lower bound is a discard strategy, not an impossibility proof.
\end{table}
'''
    (ROOT/'draft/rate_table.tex').write_text(text,encoding='utf-8')
    # Representative recursive schedules; exact IDs and allocations accompany the CSV.
    protocols=load('representative_protocols.json')['protocols'];selected=[]
    for model,q,M in [('Werner',.28,4),('Werner',.28,8),('Werner',.3,2),('Werner',.3,4),('Werner',.3,8),('Werner',.32,8),('BB84',.2,2),('BB84',.22,2),('BB84',.22,4),('BB84',.22,8),('BB84',.24,4),('BB84',.24,8)]:
        permission='full' if model=='Werner' else 'real'
        v=next(r for r in protocols if r['model']==model and r['permission']==permission and r.get('Q',r.get('b'))==q and r['budget']==M)
        selected.append(v)
    codes={e['id']:e for e in json.loads((EXT/'data/codes.json').read_text())['entries']}
    text=r'''\begin{table}[t]
\centering\small
\caption{Executable representative schedules. The register column gives the supplied implementation cost, which can be below the budget where the protocol is selected. Words are read from inputs to output. $m_s$ counts inputs on a successful ancestry, not the average cost. $C$ includes all ED retries, before the classical B prefix. For BB84, $C$ depends on the unknown correlation and is not replaced by a single number. Every row except the compressed-OTP row uses no OTP.}\label{tab:protocols}
\begin{tabular}{lccccccr}
\toprule Input/$Q$ & ED and final axis & $M$ & $m_s$ & $B$ & Terminal & Mode & $C$\\\midrule
'''
    for v in selected:
        ed=v['ED'];pump=ed['family']=='real_decoder_pumping';word=(ed['primitive']+f"-pump({ed['r']})") if pump else ed['path']
        ancestors=ed['r'] if pump else 2**len(ed['path'])
        a=v.get('AD') or v['strategy'][0]['AD'];B=a.get('classical_B_rounds',v.get('B',0));mode='comp.' if a.get('mode')=='otp_compressed' else '--'
        if a['kind']=='Watanabe_2007':terminal='W'
        elif a['kind']=='SPC':terminal=f"SPC {a['n']}"
        elif a['kind']=='linear':
            c=codes[a['code']];terminal=f"SPC {c['n']}" if c['k']==c['n']-1 else f"$[{c['n']},{c['k']},{c['d']}]$"
        else:terminal=f"rep. {a['n']}"
        C=f"{float(v['ED_cost_interval'][0]):.2f}" if v.get('ED_cost_interval') else '--'
        q=v.get('Q',v.get('b'))
        text+=f"{v['model']} / {q:.2f} & $\\mathtt{{{word}}}/{ed['final_axis']}$ & {ed['memory_upper_bound']} & {ancestors} & {B} & {terminal} & {mode} & {C}"+r'\\'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\par\smallskip\footnotesize W denotes Eq.~\eqref{eq:watanabe}. SPC $n$ has generator $[I_{n-1}\mid\boldsymbol1]$. The JSON records give the other generators, code allocations and interval bounds.
\end{table}
'''
    (ROOT/'draft/protocol_table.tex').write_text(text,encoding='utf-8')
    text=r'''\begin{table}[t]
\centering\small
\caption{Separate enlarged-axis BB84 comparison, permitting final $X/Y/Z$ measurements. These intervals cover the registered recursive/pumping menu, not the complete-block union. Lower rates are rounded down; each gap bounds the difference from the corresponding menu upper certificate. This permission is different from the main $X/Z$ comparison.}\label{tab:full-permission}
\begin{tabular}{cccccc}\toprule $Q=b=p$ & $M\leq1$ & $M\leq2$ & $M\leq3$ & $M\leq8$ & Max. gap\\\midrule
'''
    for q in (.15,.2,.22,.24):
        src=load(f'bb84_joint_extension_full_{q:.5f}_{q:.5f}_n128_menu_certified.json')
        if not src or not src.get('complete'):continue
        r=src['results'];budgets=(1,2,3,8)
        text+=f'{q:.2f} & '+' & '.join(lower_tex(r[f'M{m}']['lower']) for m in budgets)+' & '+upper_tex(max(r[f'M{m}']['gap_upper'] for m in budgets))+r'\\'+'\n'
    text+=r'''\bottomrule\end{tabular}
\par\smallskip\footnotesize A zero lower entry is the discard guarantee at the stated numerical precision, not a zero-capacity claim. The $Q=0.24$ small-register interval is relatively loose; the analytic positivity statement is separate.
\end{table}'''+ '\n'
    (ROOT/'draft/full_permission_table.tex').write_text(text,encoding='utf-8')
    print('Union tables:',len(union),'certified budget/input entries.',flush=True)


if __name__=='__main__':main()
