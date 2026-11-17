"""One source of truth for exact resources and honest result categories."""
from common import *
import json,csv,math,hashlib
from pathlib import Path


def read(p):return json.loads(p.read_text())
def protocol_key(r):return hashlib.sha256(json.dumps([r['Q'],r['ED'],r['branches']],sort_keys=True).encode()).hexdigest()[:20]


def main():
    small=read(DATA/'small_memory_exact.json')['entries'];graph=read(DATA/'graph_memory_exact.json')
    old=read(EXT/'data/combined_results.json');oldcat=read(EXT/'data/protocol_catalog.json')
    fixed=read(DATA/'fixed_pumping_certified.json')['results'] if (DATA/'fixed_pumping_certified.json').exists() else {}
    certs=read(DATA/'envelope_werner_certificates.json')['results'] if (DATA/'envelope_werner_certificates.json').exists() else {}
    # Preserve old files as historical snapshots, replace only current costs.
    def memory(ed,fallback=2):
        if ed in small:return small[ed]['M']
        gid=ed.split('_')[0] if ed.startswith('XG') else ed
        if gid in graph['entry_to_graph']:return graph['graphs'][graph['entry_to_graph'][gid]]['M']
        if ed=='restart_pump_1':return 1
        if ed.startswith(('pump_','restart_pump')):return 2
        if ed.startswith('direct_'):return 1
        return fallback
    candidates_six={};candidates_bb={};catalog={};sixrows=[];bbrows=[]
    for pid,record in old['protocols'].items():
        r={**record,'source_snapshot':pid,'category':'additional_achievable_construction'}
        if pid.startswith('SIX'):
            r['M']=memory(r['ED'],r['M']);candidates_six.setdefault(r['Q'],[]).append(r)
        else:
            fk=f"B_{r['eb']:.5f}_{r['ep']:.5f}_real_M2"
            if r.get('fixed_protocol') and fk in fixed and r['fixed_protocol']['ED']==fixed[fk]['fixed_protocol']['ED']:
                r['rate']=float(fixed[fk]['lower']);r['certificate']='fixed_pumping_certified.json';r['certified_lower']=fixed[fk]['lower']
            r['rate']=max(0.,math.nextafter(r['rate'],-math.inf))
            producer=max((memory(e,r['M']) for e in r['EDs']),default=r['M'])
            r['M']=max(producer,2 if r['permission']=='real' else 1)
            candidates_bb.setdefault((r['eb'],r['ep']),[]).append(r)
    finite=read(DATA/'werner_complete_k16.json')
    for row in finite['results']:
        candidates_six.setdefault(row['Q'],[]).extend({**r,'source':'werner_complete_k16.json','category':'complete_quantum_short_code_numerical'} for r in row['results'].values())
    for longfile in sorted(DATA.glob('werner_complete_long*.json')):
        for row in read(longfile)['results']:
            candidates_six.setdefault(row['Q'],[]).extend({**r,'source':longfile.name,'category':'complete_quantum_long_endpoint_numerical'} for r in row['results'].values())
    if (DATA/'larger_werner.json').exists():
        for row in read(DATA/'larger_werner.json')['results']:
            candidates_six.setdefault(row['Q'],[]).extend({**r,'source':'larger_werner.json','category':'larger_constructed_family'} for r in row['results'].values())
    if (DATA/'pumping_fair_codes.json').exists():
        for row in read(DATA/'pumping_fair_codes.json')['results']:
            candidates_six.setdefault(row['Q'],[]).extend({**r,'source':'pumping_fair_codes.json','category':'restart_pumping_matched_AD'} for r in row['results'].values())
    for path in sorted(DATA.glob('bb84_complete_*_certified.json')):
        c=read(path);srcpath=DATA/path.name.replace('_certified','');src=read(srcpath);eb=src['eb'];ep=src['ep']
        for mode,r in c['results'].items():
            permission,M=mode.split('_M');policy=src['results'][mode]['policy'];ids=sorted({v['ED'] for v in policy})
            maxm=max((int(small[e]['measurement_key'].split(':')[0]) for e in ids),default=1)
            cand=dict(rate=max(0.,math.nextafter(float(r['lower']),-math.inf)),permission=permission,M=int(M),m_max=maxm,EDs=ids,policy=policy,
                      source=path.name,category='complete_small_class_certified_policy',eb=eb,ep=ep)
            candidates_bb.setdefault((eb,ep),[]).append(cand)
    codes={e['id']:e for e in read(OLD/'data/codes.json')['entries']}
    codes.update({e['id']:e for e in read(EXT/'data/codes.json')['entries']})
    def code_text(b):
        if b.get('kind','linear')=='linear':
            e=codes[b['code']];return f"[{e['n']},{e['k']},{e['d']}]"
        n=b['n'];return f'[{n},1,{n}]' if b['kind']=='repetition' else f'[{n},{n-1},2]'
    for q,choices in sorted(candidates_six.items()):
        for M in range(1,9):
            eligible=[r for r in choices if r['M']<=M];maximum=max(r['rate'] for r in eligible)
            tied=[r for r in eligible if r['rate']>=maximum*(1-1e-10)]
            def complexity(r):
                return sum((2**b['binary_rounds'])*(codes[b['code']]['n'] if b.get('kind','linear')=='linear' and b.get('code') else b.get('n',1)) for b in r['branches'] if b['contribution']>0)
            best=min(tied,key=lambda r:(r['M'],r['m'],complexity(r),not r['ED'].startswith('LC'),r['ED']));pid=f'W_{q:.5f}_M{M}'
            best=best.copy();key=protocol_key(best)
            if key in certs:
                best.update(numerical_rate=best['rate'],rate=max(0.,math.nextafter(float(certs[key]['lower']),-math.inf)),certified_lower=certs[key]['lower'],certificate_id=key)
            summary='; '.join(f"s={b['syndrome']}: B^{b['binary_rounds']} + {code_text(b)}" for b in best['branches'] if b['contribution']>0 and b.get('code',True))
            catalog[pid]={**best,'budget':M,'AD_summary':summary}
            sixrows.append(dict(Q=q,budget=M,memory=best['M'],inputs=best['m'],rate=best['rate'],ED=best['ED'],AD=summary,category=best['category'],protocol=pid))
    for (eb,ep),choices in sorted(candidates_bb.items()):
        for permission in ('real','full'):
            for M in range(2 if permission=='real' else 1,9):
                eligible=[r for r in choices if r['M']<=M and (permission=='full' or r['permission']=='real')]
                if not eligible:continue
                best=max(eligible,key=lambda r:r['rate']);pid=f'B_{eb:.5f}_{ep:.5f}_{permission}_M{M}'
                catalog[pid]={**best,'budget':M,'reported_permission':permission}
                bbrows.append(dict(eb=eb,ep=ep,permissions=permission,budget=M,memory=best['M'],max_inputs=best['m_max'],lower=best['rate'],category=best['category'],protocol=pid))
    # A class optimum enclosure is kept separate from the envelope above.
    bbcore=[]
    for path in sorted(DATA.glob('bb84_complete_*_certified.json')):
        c=read(path);src=read(DATA/path.name.replace('_certified',''));ubpath=DATA/path.name.replace('bb84_complete_','bb84_upper_').replace('_certified','')
        upper=read(ubpath)['results'] if ubpath.exists() else {}
        for mode,r in c['results'].items():
            bbcore.append(dict(eb=src['eb'],ep=src['ep'],mode=mode,lower=float(r['lower']),
                               numerical_upper=src['results'][mode]['upper'],certified_upper=upper.get(mode,{}).get('upper'),
                               source=path.name))
    wcore=[]
    for path in sorted(DATA.glob('werner_upper_*.json')):
        src=read(path)
        from fractions import Fraction
        for mode,r in src['results'].items():wcore.append(dict(Q=float(Fraction(src['Q'])),mode=mode,**r))
    combined=dict(scope='Achievable envelopes with exact implementation memory, finite-class results separately identified. Discrete physical error points. BB84 policy lower bounds cover all compatible t.',
                  six_state=sixrows,bb84=bbrows,bb84_complete_core=bbcore,werner_complete_core=wcore,protocols=catalog)
    (DATA/'current_results.json').write_text(json.dumps(combined,separators=(',',':')))
    for name,rows in [('six_state',sixrows),('bb84',bbrows),('bb84_complete_core',bbcore),('werner_complete_core',wcore)]:
        with (DATA/f'{name}_summary.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert all(math.isfinite(r['rate']) and r['rate']>=0 and r['memory']<=r['budget'] for r in sixrows)
    assert all(math.isfinite(r['lower']) and r['lower']>=0 and r['memory']<=r['budget'] for r in bbrows)
    print('Assembled',len(sixrows),'Werner rows,',len(bbrows),'BB84 rows,',len(bbcore),'finite BB84 core enclosures',flush=True)


if __name__=='__main__':main()
