"""Assemble traceable tables without promoting searches to certificates."""
from core import *
import csv,hashlib


def read(name):
    path=DATA/name
    return json.loads(path.read_text()) if path.exists() else None


def collect():
    files={};tables={};protocols=[]
    def use(name):
        obj=read(name)
        if obj is not None:files[name]=hashlib.sha256((DATA/name).read_bytes()).hexdigest()
        return obj
    werner=[]
    for permission in ('real','full'):
        src=use(f'werner_v2_{permission}_menu_certified.json')
        if src:
            witness=use(f'werner_v2_{permission}_certified.json')
            for item in src['results'].values():
                q=item['Q']
                for budget,row in item['results'].items():
                    lo=float(row['lower']);hi=float(row['upper'])
                    werner.append(dict(permission=permission,Q=q,M=int(budget[1:]),lower=lo,upper=hi,gap=hi-lo,
                                       scope='registered_recurrence_and_AD'))
                    key=f'{q:.12g}_{budget}';w=witness['results'][key]
                    protocols.append(dict(model='Werner',permission=permission,Q=q,budget=int(budget[1:]),
                                          ED=w['ED'],AD=w['AD'],lower=lo,upper=hi,
                                          ED_cost_interval=w.get('ED_cost_interval'),source=f'werner_v2_{permission}_certified.json'))
    tables['werner_recurrence']=sorted(werner,key=lambda r:(r['permission'],r['Q'],r['M']))
    bb=[]
    for path in sorted(DATA.glob('bb84_joint_extension_*_n128_menu_certified.json')):
        src=use(path.name)
        if not src.get('complete'):continue
        source=use(path.name.replace('_menu_certified',''))
        for budget,row in src['results'].items():
            bb.append(dict(permission=src['permissions'],b=src['b'],p=src['p'],M=int(budget[1:]),
                           lower=row['lower'],upper=row['upper'],gap=row['gap_upper'],converged=row['converged'],
                           scope='registered_recurrence_and_fixed_AD_mixtures'))
            witness=source['results'][budget]
            protocols.append(dict(model='BB84',permission=src['permissions'],b=src['b'],p=src['p'],budget=int(budget[1:]),
                                  lower=row['lower'],upper=row['upper'],ED=witness.get('ED'),
                                  B=witness.get('classical_B_rounds',witness.get('AD',{}).get('classical_B_rounds')),
                                  strategy=witness.get('strategy'),AD=witness.get('AD'),source=path.name.replace('_menu_certified','')))
    tables['bb84_recurrence']=bb
    for name,rows in tables.items():
        if not rows:continue
        with (DATA/f'table_{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    codes={e['id']:e for e in json.loads((EXT/'data/codes.json').read_text())['entries']}
    for row in protocols:
        ads=[row['AD']] if row.get('AD') else [item['AD'] for item in row.get('strategy') or []]
        row['short_code_definitions']={a['code']:codes[a['code']] for a in ads if a and a.get('kind')=='linear'}
    atomic_json(DATA/'representative_protocols.json',dict(protocols=protocols,
                  decoder_source='recurrence_v2_checks.json',word_order='raw inputs to root',
                  code_convention='classical binary [n,k,d]; SPC generator [I_(n-1) | 1], repetition generator all-ones'))
    states={}
    for name in ('werner_complete_joint_n128.json','werner_joint_union_core_certified.json',
                 'bb84_policy_curves_real.json','bb84_policy_curves_full.json','bb84_asymmetric_grid.json'):
        src=use(name)
        if src:states[name]=dict(saved_results=len(src.get('results',[])),scope=src.get('scope'))
    out=dict(file_sha256=files,tables={k:len(v) for k,v in tables.items()},other_states=states,
             note='Snapshot only. A missing certificate remains missing; selected-policy bounds are not converted to class optima. Running jobs can supersede this snapshot.')
    atomic_json(DATA/'result_manifest.json',out);print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':collect()
