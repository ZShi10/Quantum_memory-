from common import *
import json,csv,hashlib,ast,math


def read(path):return json.loads(path.read_text())


def main():
    memory_data=read(DATA/'memory_refinement.json');memory=memory_data['results'];small_full=memory_data['small_full'];six=read(DATA/'six_state.json')
    oldsix=read(OLD/'data/werner_joint_k12f.json')['results'];pumps=read(OLD/'data/werner_pumping_joint.json')['results']
    precise={(r['Q'],r['ED']):float(r['rate_130digit']) for r in read(DATA/'witnesses_130digit.json')['results']}
    shortcodes={e['id']:e for e in read(DATA/'codes.json')['entries']}
    oldcodes={e['id']:e for e in read(OLD/'data/codes.json')['entries']};codes={**oldcodes,**shortcodes}
    oldblocks={e['id']:e for e in read(OLD/'data/blocks.json')['entries']}
    catalog={};sixrows=[]
    def code_description(branch):
        kind=branch.get('kind','linear')
        if kind=='linear':
            c=codes.get(branch.get('code'))
            return f"[{c['n']},{c['k']},{c['d']}]" if c else 'discard'
        n=branch['n'];return f'[{n},1,{n}]' if kind=='repetition' else f'[{n},{n-1},2]'
    for item in six['results']:
        q=item['Q'];choices=[]
        for budget,r in item['results'].items():
            r={**r};e=six['protocol_library'][r['ED']]
            if r['ED'] in memory:r['M']=memory[r['ED']]['M_full']
            elif r['ED'] in small_full:r['M']=small_full[r['ED']]['M_full']
            r['rate']=precise.get((q,r['ED']),r['rate']);r['source']='six_state.json'
            choices.append(r)
        for mode,records in oldsix.items():
            if not mode.startswith('full'):continue
            for r in records:
                if abs(q-r['Q'])<1e-10:
                    r={**r,'source':'../joint_optimization/data/werner_joint_k12f.json'}
                    if r['ED'] in memory:r['M']=memory[r['ED']]['M_full']
                    elif r['ED'] in small_full:r['M']=small_full[r['ED']]['M_full']
                    choices.append(r)
        for r in pumps:
            if abs(q-r['Q'])<1e-10:
                # In this saved scan no longer-repetition alternative exceeds
                # the explicit short-code witness by a meaningful margin.
                assert r['combined_rate']<=r['rate']*(1+1e-6)+1e-18
                choices.append(dict(Q=q,ED=f"restart_pump_{r['r']}",m=r['r'],M=r['M'],rate=r['rate'],
                                    source='../joint_optimization/data/werner_pumping_joint.json',expected_raw_cost=r['expected_raw_cost'],
                                    branches=[dict(syndrome=0,code=r['code'],binary_rounds=r['binary_rounds'],contribution=r['rate'])]))
        for M in range(1,9):
            best=max((r for r in choices if r['M']<=M),key=lambda r:r['rate'])
            pid=f'SIX_{q:.5f}_M{M}';record={**best,'budget':M}
            record['AD_summary']='; '.join(f"s={b['syndrome']}: B^{b['binary_rounds']} + {code_description(b)}" for b in best['branches'] if b['contribution']>0)
            for b in record['branches']:
                if b.get('code') in codes:b['generator_rows']=codes[b['code']]['rows']
            catalog[pid]=record;sixrows.append(dict(Q=q,M_budget=M,M_used=best['M'],m=best['m'],rate=best['rate'],protocol=pid,ED=best['ED'],AD=record['AD_summary']))
    oldcert={r['source']:r for r in read(OLD/'data/all_policy_interval_certificates.json')['results']}
    longdata={M:{r['Q']:r for r in read(DATA/f'bb84_long_repetition_M{M}_certified.json')['results']} for M in (1,2)}
    longreal={r['Q']:r for r in read(DATA/'bb84_long_repetition_M2_real_certified.json')['results']}
    bbrows=[];rules_count=0;cells_count=0
    for path in sorted(DATA.glob('bb84_[0-9]*.json')):
        src=read(path);eb,ep=src['eb'],src['ep'];cert=read(DATA/path.name.replace('bb84_','bb84_certified_'))
        candidates=[]
        for mode,r in cert['results'].items():
            permission=mode.split('_')[0];used=0;used_full=0;mmax=1
            for cell in r['policy']:
                e=src['protocol_library'][cell['ED']];cost=e['M'];mmax=max(mmax,e['m'])
                fullcost=memory[e['graph_id']]['M_full'] if e['source']=='extended_graph' else small_full.get(e['id'],dict(M_full=min(2,e['M'])))['M_full']
                used_full=max(used_full,fullcost)
                if permission=='full':cost=fullcost
                used=max(used,cost)
            candidates.append(dict(rate=float(r['lower']),permission=permission,M=max(used,2 if permission=='real' else 1),m_max=mmax,
                                   source=cert['source'].replace('bb84_','bb84_certified_'),mode=mode,
                                   EDs=sorted({c['ED'] for c in r['policy']}),policy=r['policy'],certificate='60-digit outward rounding'))
            if permission=='real':candidates.append({**candidates[-1],'permission':'full','M':max(used_full,1)})
            if float(r['lower'])>0:rules_count+=1;cells_count+=len(r['policy'])
        oldname=f'bb84_pilot_{eb:.5f}_{ep:.5f}.json'
        if oldname in oldcert:
            for mode,r in oldcert[oldname]['results'].items():
                if 'lower' not in r or not r.get('cells'):continue
                policy=r['cells'];used=max(max(c['M'] for c in policy),2 if mode.startswith('real') else 1)
                candidates.append(dict(rate=float(r['lower']),permission=mode.split('_')[0],M=used,m_max=max(oldblocks[c['ED']]['m'] for c in policy),
                                       source='../joint_optimization/data/all_policy_interval_certificates.json',mode=mode,source_result=oldname,
                                       EDs=sorted({c['ED'] for c in policy}),policy=policy,certificate='60-digit outward rounding'))
                if mode.startswith('real'):
                    fullcost=max(small_full.get(c['ED'],dict(M_full=min(2,c['M'])))['M_full'] for c in policy)
                    candidates.append({**candidates[-1],'permission':'full','M':fullcost})
        if eb==ep:
            for M in (1,2):
                if eb not in longdata[M]:continue
                r=longdata[M][eb]
                candidates.append(dict(rate=float(r['lower']),permission='full',M=M,
                                       m_max=max(c['parameters'].get('r',1) for c in r['policy']),
                                       source=f'bb84_long_repetition_M{M}_certified.json',mode=f'full_M{M}',
                                       EDs=sorted({c['ED'] for c in r['policy']}),policy=r['policy'],certificate='60-digit outward rounding'))
            if eb in longreal and float(longreal[eb]['lower'])>0:
                r=longreal[eb]
                candidates.append(dict(rate=float(r['lower']),permission='real',M=2,
                                       m_max=max(c['parameters'].get('r',1) for c in r['policy']),
                                       source='bb84_long_repetition_M2_real_certified.json',mode='real_M2',
                                       EDs=sorted({c['ED'] for c in r['policy']}),policy=r['policy'],certificate='60-digit outward rounding'))
        streaming=OLD/'data'/f'streaming_{eb:.5f}_{ep:.5f}.json'
        if streaming.exists():
            r=read(streaming)['results']['real_M2']
            if r['lower']>0:candidates.append(dict(rate=r['lower'],permission='real',M=2,m_max=r['parameters'].get('r',1),source='../joint_optimization/data/'+streaming.name,
                                                  mode='real_M2',EDs=[r['ED']],fixed_protocol=r,certificate='Previous continuous numerical enclosure'))
        for permission in ('real','full'):
            for M in range(2 if permission=='real' else 1,9):
                eligible=[r for r in candidates if r['M']<=M and (permission=='full' or r['permission']=='real')]
                best=max(eligible,key=lambda r:r['rate']);pid=f'BB_{eb:.5f}_{ep:.5f}_{permission}_M{M}'
                catalog[pid]={**best,'eb':eb,'ep':ep,'budget':M}
                bbrows.append(dict(eb=eb,ep=ep,permissions=permission,M_budget=M,M_used=best['M'],m_max=best['m_max'],rate=best['rate'],protocol=pid,EDs=';'.join(best['EDs'])))
    for name,rows in (('six_state_summary.csv',sixrows),('bb84_summary.csv',bbrows)):
        with (DATA/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(scope='Best achieved envelope over saved new and previous constructions, with full-Clifford memory refinements. Parameter-grid summaries, not certified Q switching intervals or global optimality.',
                six_state=sixrows,bb84=bbrows,protocols=catalog)
    (DATA/'combined_results.json').write_text(json.dumps(result,separators=(',',':')))
    audit=dict(status='passed',new_graph_candidates=len(read(DATA/'graphs.json')['entries']),new_code_entries=len(read(DATA/'codes.json')['entries']),
               six_state_error_points=len({r['Q'] for r in sixrows}),bb84_error_points=len({(r['eb'],r['ep']) for r in bbrows}),
               new_positive_BB84_feedback_rules=rules_count,new_certified_BB84_cells=cells_count,
               representative_six_state_high_precision_witnesses=len(precise),memory_refinements=len(memory),
               explicit_small_full_compilations=len(small_full),certified_long_repetition_rules=sum(len(v) for v in longdata.values())+sum(float(r['lower'])>0 for r in longreal.values()))
    for name in ('character_verification.json','rate_verification.json','witnesses_130digit.json','memory_refinement.json'):
        assert read(DATA/name)['status']=='passed'
    for M in (1,2):assert read(DATA/f'bb84_long_repetition_M{M}_certified.json')['status']=='passed'
    assert read(DATA/'bb84_long_repetition_M2_real_certified.json')['status']=='passed'
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    for rows,group in ((sixrows,lambda r:(r['Q'],)),(bbrows,lambda r:(r['eb'],r['ep'],r['permissions']))):
        groups={}
        for r in rows:
            assert math.isfinite(r['rate']) and r['rate']>=0 and r['M_used']<=r['M_budget']
            groups.setdefault(group(r),[]).append(r)
        for values in groups.values():
            ordered=sorted(values,key=lambda r:r['M_budget'])
            assert all(b['rate']>=a['rate']-1e-15 for a,b in zip(ordered,ordered[1:]))
    (DATA/'validation_summary.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
