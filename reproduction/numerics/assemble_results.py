"""Assemble fresh certificates and clearly labelled inherited endpoints.

No optimizer is run here. Exact rational endpoint arithmetic determines
target-memory windows; decimal conversions are only used for display.
"""
from pathlib import Path
from fractions import Fraction as F
import csv
import hashlib
import json
import sys
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction/extensions/code'))
from bell_core import homogeneous,axis,watanabe,entropy
from verify import terminal_mem

def read(p): return json.loads((ROOT/p).read_bytes())
def dump(p,obj):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def csvout(p,rows):
    path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def upward(x):
    y=float(x)
    return math.nextafter(y,math.inf) if F.from_float(y)<x else y
def texnum(x,significant=7):
    mant,exp=f'{x:.{significant-1}e}'.split('e')
    return mant.rstrip('0').rstrip('.')+r'\times10^{'+str(int(exp))+'}'

def main():
    rows=list(csv.DictReader((ROOT/'reproduction/controls/data/fixed_block_interval_bounds.csv').open()))
    bounds={(int(r['Nmax']),int(r['memory'])):r for r in rows}
    check=read('reproduction/controls/verification/output_consistency.json')
    mp={r['memory']:r['rate'] for r in check['independent_high_precision_winner_rates']}
    controls=[]
    for w in read('reproduction/controls/data/executable_control_winners.json'):
        m=w['peak_memory'];b=bounds[6,m];b8=bounds[8,m]
        assert b['lower']==b8['lower'] and b['upper']==b8['upper']
        C=F(w['mean_raw_cost']);r=float(mp[m]);q=[float(F(v)) for v in w['conditioned_Bell_probabilities']]
        controls.append(dict(source_model='Werner',Q='7/25',ED_family='complete graph/dual zero-syndrome control',
            N=6,memory_budget=m,graph=w['graph'],logical_dual=w['logical_dual'],
            order=' '.join(map(str,w['input_order'])),retry_rule='fixed-order eager rejection; independent full restart',
            classical='two pooled B rounds + Watanabe; no AD masks',
            C_exact=str(C),C=float(C),p_accept=float(F(w['acceptance_probability'])),
            r_cl=r*float(C),output_fidelity=max(q),
            rate=r,rate_100digits=mp[m],lower=b['lower'],upper=b['upper'],
            upper_scope='all N<=8, all duals and legal fixed orders; B<=16 + Watanabe',
            result_kind='fresh exhaustive interval enclosure',
            evidence='reproduction/controls/data/executable_control_winners.json'))
    csvout('data/figure3a_matched_control.csv',controls)
    gains=dict(ratio_lower=str(F(float(bounds[6,4]['lower']))/F(float(bounds[8,3]['upper']))),
               absolute_gap_lower=str(F(float(bounds[6,4]['lower']))-F(float(bounds[8,3]['upper']))))
    q=[F(277,346),F(61,346),F(4,346),F(4,346)];rcl=1-entropy(q)
    same=[]
    for m,C,order in [(2,F(2975,173),'3 2 1 4 0'),(3,F(11425,692),'3 2 4 0 1')]:
        same.append(dict(source_model='Werner',Q='1/5',N=5,memory_budget=m,
            adjacency='18 5 10 4 1',logical_dual=5,order=order,
            retry_rule='same fixed-order eager stopping; order optimized within memory',
            classical='same one-way 1-H terminal',C_exact=str(C),C=float(C),
            r_cl=rcl,rate=rcl/float(C),p_accept='692/3125',
            output_Bell='277/346 61/346 4/346 4/346',status='exact cost; floating entropy only for display'))
    assert F(same[0]['C_exact'])/F(same[1]['C_exact'])==F(476,457)
    csvout('data/figure3b_same_measurement.csv',same)
    curves=read('data/inherited/frozen_curve_delivery.json')['points']
    index={(p['model'],F(str(p['Q']))):p for p in curves}
    lower=[];windows=[];samples=[]
    targets={'BB23_m5':'0.0005','BB23_m6':'0.0008','BB24_m6':'0.00001',
             'BB24_m7':'0.00008','BB24_m8':'0.000095'}
    for path in sorted((ROOT/'reproduction/extensions/data').glob('*_lower_certificate.json')):
        cert=json.loads(path.read_bytes());spec=cert['specification'];Q=F(spec['Q'])
        m=terminal_mem(spec['word'],spec['axis']);L=F(spec['target'])
        assert cert['status']=='verified_continuous_fixed_policy_lower'
        assert all(F(leaf['lower'])>=L for leaf in cert['leaves'])
        row=dict(policy=cert['policy'],source_model='BB84_real',Q=str(Q),N=2**len(spec['word']),
                 memory=m,word=spec['word'],axis=spec['axis'],classical='Watanabe, B=0, no AD masks',
                 retry_rule='retain independent child success and retry sibling; registered terminal replacement',
                 lower_exact=str(L),lower=float(L),interval_leaves=len(cert['leaves']),
                 status='fresh full-z fixed-policy lower certificate',certificate=str(path.relative_to(ROOT)).replace('\\','/'))
        lower.append(row)
        if cert['policy'] in targets:
            p=index['BB84_real',Q]
            U0=F.from_float(p['results'][f'M{m-1}']['upper'])
            U3=F.from_float(p['results']['M3']['upper'])
            terms={'finite_upper':U0,'stopping_upper':5*U3,'B_tail':F(1,2**17),'pump_tail':(1-Q)**160/2}
            U=max(terms.values());T=F(targets[cert['policy']]);assert U<T<=L
            win=dict(Q=str(Q),memory=m,upper_low_exact=str(U),lower_high_exact=str(L),
                     upper_low=upward(U),lower_high=float(L),target=float(T),target_exact=str(T),
                     ratio_lower_exact=str(L/U),ratio_lower=float(L/U),policy=cert['policy'],
                     upper_source='inherited finite upper + proved enlargement; exact rational arithmetic',
                     lower_source='fresh full-z fixed-policy certificate',
                     upper_active_term=next(k for k,v in terms.items() if v==U),
                     interval='(upper_low, lower_high]',family='P_ext; original fixed-mode/fixed-allocation coefficients')
            windows.append(win)
        for t in np.linspace(0,1,501):
            z=float(Q)*t;p=[1-2*float(Q)+z,float(Q)-z,z,float(Q)-z]
            out,C=homogeneous(p,spec['word']);out=axis(out,spec['axis']);rc=watanabe(out)
            samples.append(dict(policy=cert['policy'],Q=float(Q),z=z,t=t,N=row['N'],memory=m,
                    C=C,r_cl=rc,rate=rc/C,output_fidelity=max(out),
                    pI=out[0],pX=out[1],pY=out[2],pZ=out[3],
                    status='floating diagnostic samples; use separate interval certificate for lower bound'))
    csvout('data/bb84_certified_policies.csv',lower)
    csvout('data/figure4_target_windows.csv',windows)
    csvout('data/bb84_policy_decomposition.csv',samples)
    dump('data/results_summary.json',dict(matched_gain=gains,matched_noise_interval=read('validation/matched_noise_interval.json')['certificate'],
        target_windows=windows,selected_lower_certificates=lower,
        graph_coverage=read('validation/graph_catalog_full.json')['coverage']))
    # Compact tables contain only transparent display rounding, never new bounds.
    tex=ROOT/'paper/generated';tex.mkdir(exist_ok=True)
    lines=[r'\begin{tabular}{rrrrlll}',r'\toprule',r'$Q$ & $m$ & $N$ & certified $L_m$ & word & axis & leaves\\\midrule']
    for r in lower:
        lines.append(f"{float(F(r['Q'])):.2f} & {r['memory']} & {r['N']} & ${texnum(r['lower'])}$ & \\texttt{{{r['word']}}} & ${r['axis']}$ & {r['interval_leaves']}\\\\")
    lines += [r'\bottomrule',r'\end{tabular}']
    (tex/'lower_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{rrrrr}',r'\toprule',r'$Q$ & $m$ & upper on $R_{m-1}^{\rm ext}$ & achievable $L_m$ & target $R_*$\\\midrule']
    for r in windows:
        # Eight significant digits rounded upward for displayed upper.
        U=F(r['upper_low_exact']);exp=math.floor(math.log10(float(U)));scale=10**(7-exp)
        num=(U.numerator*scale+U.denominator-1)//U.denominator;ud=num/scale
        lines.append(f"{float(F(r['Q'])):.2f} & {r['memory']} & ${texnum(ud,8)}$ & ${texnum(r['lower_high'])}$ & ${texnum(r['target'])}$\\\\")
    lines += [r'\bottomrule',r'\end{tabular}']
    (tex/'target_table.tex').write_text('\n'.join(lines)+'\n')
    claims=[
      ('C1','Two local qubits suffice for uniform strictly positive asymptotic key below the stated boundaries.',
       'analytic proof; gate and pumping identities checked','Known entangled Bell laws; BB84 eb+ep<1/2; IID ideal finite construction; not universal memory necessity',
       ['paper/sections/s3_production.tex','reproduction/extensions/validation/pumping_checks_results.json']),
      ('C2','Exact minimum width of the specified complete labelled Clifford readout.',
       'analytic proof + fresh exhaustive structural checks','Complete rank-one measurement; counted input port; no external quantum buffer',
       ['paper/sections/s2_measurement.tex','validation/enumeration.json','data/measurement_schedules.json']),
      ('C3','All graph LC/permutation classes through N=8 and real bases through N=5 covered.',
       'fresh independent exhaustive enumeration','Measurement bases; not arbitrary quantum-output instruments',
       ['validation/graph_catalog_full.json','validation/enumeration.json']),
      ('C4','All binary linear-code permutation classes through length 8 covered (612).',
       'fresh independent exhaustive enumeration','Finite short-code menu only; no unrestricted repetition/SPC optimality theorem',
       ['validation/enumeration.json','data/codes_complete_n8.json']),
      ('C5','Werner Q=0.28 matched four-slot control exceeds every <=3-slot control with <=8 inputs by >4.09%.',
       'fresh whole-control interval evaluation; 100-digit witness cross-check',
       'Zero syndrome; all fixed eager orders; B<=16 + Watanabe; one private output; all plotted winners N=6',
       ['reproduction/controls/verification/fixed_block_interval_certificate.json','data/figure3a_matched_control.csv']),
      ('C6','Strict matched advantage throughout Q in [0.279999,0.280001].',
       'fresh outward interval certificate; no interpolation','Same control as C5; one constant N=6,m=4 witness throughout',
       ['validation/matched_noise_interval.json','numerics/matched_interval.py']),
      ('C7','Same five-input measurement gains rate factor 476/457 by lower consumption at three slots.',
       'exact rational costs; all 120 orders checked','Fixed same measurement, outcome, classical one-way terminal and eager-stopping rule; only legal loading order changes',
       ['data/figure3b_same_measurement.csv','reproduction/extensions/validation/independent_checks.json']),
      ('C8','Global Werner and BB84 rate landscapes reproduce supplied tables exactly.',
       'inherited search evidence; all archive hashes and both 1528-row tables rechecked',
       'Finite P0 with full-block costs; 71 BB84 and 6 Werner certified anchors; 114 Werner floating-search points; no new whole-class optimizer rerun',
       ['validation/legacy_audit.json','data/global_rate_landscape.csv','data/inherited/frozen_curve_delivery.json']),
      ('C9','Six selected BB84 policies achieve their lower rates for every compatible z.',
       'fresh interval generation and full leaf replay','Fixed word and X/Z axis; no masks or B prefix; all z in [0,Q]',
       ['data/bb84_certified_policies.csv','reproduction/extensions/data/BB23_m5_lower_certificate.json','validation/runs/theory-checks.log']),
      ('C10','Specified target yields require memories 5,6 at Q=.23 and 6,7,8 at Q=.24.',
       'fresh lower certificates + inherited finite upper enclosures + analytic enlargement',
       'P_ext union and registered memory schedules; original fixed-mode coefficients; not the stronger public-hashed-max BB84 objective',
       ['paper/sections/s6_enlarged.tex','data/figure4_target_windows.csv']),
      ('C11','One-way terminal raw pumping has a separately scoped symmetric-extension obstruction.',
       'analytic closure + exhaustive two-input normal-form checks','Retained-output two-slot raw-pumping class; no general ED+AD or CRO converse',
       ['paper/sections/s8_comparisons.tex','reproduction/extensions/validation/pumping_checks_results.json']),
      ('C12','Literature baselines retain their original computation and normalization.',
       'Sun eight author envelopes independently recomputed and nine-size plot matched exactly; Du supplied reference retained','Different ED/AD menus; contextual comparison, not a matched memory converse',
       ['paper/sections/s8_comparisons.tex','validation/literature_reconstruction.json','reproduction/extensions/data/literature/werner_linear_Sun_author.csv'])]
    ledger=[]
    for id_,statement,status,scope,evidence in claims:
        ledger.append(dict(id=id_,statement=statement,reproducibility_status=status,scope=scope,evidence=evidence))
    dump('validation/claim_ledger.json',ledger)
    csvout('validation/claim_ledger.csv',[dict(r,evidence='; '.join(r['evidence'])) for r in ledger])
    report=['VALIDATION REPORT — Quantum memory as an efficiency resource for key distillation','',
      'Rates: asymptotic net secret bits per consumed raw Bell pair. N is successful ancestry; m is peak local occupied qubits; C includes failures.',
      'Historical optimization was NOT independently rerun. The complete frozen evidence is supplied and hash-audited. Independent new calculations are listed explicitly.','']
    for r in ledger:
        report += [r['id']+'  '+r['statement'],'Status: '+r['reproducibility_status'],'Scope: '+r['scope'],
                   'Evidence: '+'; '.join(r['evidence']),'']
    report += ['Important nonclaims:',
      '- No unrestricted LOCC memory capacity or general CRO-only upper bound.',
      '- No strict improvement at every memory increment or source error.',
      '- No claim that all high-N ED circuits have been enumerated.',
      '- No broad BB84 optimality transfer to the public-hashed-max objective.',
      '- No finite-key or noisy-storage throughput result.',
      '- Blank target windows and unevaluated source points carry no optimum claim.',
      '- Bibliographic and PDF rendering checks are recorded separately in validation/.']
    (ROOT/'VALIDATION_REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    dump('validation/results_assembly.json',dict(status='passed',matched_rows=len(controls),target_windows=len(windows),
         fresh_BB84_certificates=len(lower),decomposition_samples=len(samples),claims=len(ledger),
         important='Exact upper endpoint arithmetic; target lower values taken from certificate targets, not rounded sampled minima.'))
    print(json.dumps(dict(status='passed',matched_gain=gains,target_windows=[{k:r[k] for k in ('Q','memory','upper_low','lower_high','target')} for r in windows]),indent=2))

if __name__=='__main__':main()
