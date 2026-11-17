"""Collect only traceable completed enclosures; never promote floating searches."""
from pathlib import Path
from fractions import Fraction as F
import argparse,csv,hashlib,json,math
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];ROOT=PROJECT/'revised'
OLD=PROJECT/'research/revision_20261006/recheck_20261021'

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(p):return Path(p).relative_to(PROJECT).as_posix()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def pointer(p):
    d=read(p);path=Path(d['path']);assert sha(path)==d['sha256'];return path,read(path)
def csvfile(p,rows):
    with p.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def discard_policy():
    # Zero is attained by consuming/measuring a raw pair and discarding all
    # data. This exact action supplies a witness only, never a family upper.
    return dict(lower=0.,discard_all=True,
        ED=dict(family='explicit_binary_tree',tree=dict(kind='raw'),canonical_tree='r',
            final_axis='Z',permissions='real',successful_ancestry_inputs=1,
            memory_upper_bound=1,terminal_memory=1,retained_memory=1,depth=0,
            raw_cost_model='one_raw_pair_per_discarded_record',
            memory_model='peak_local_live_qubits_including_input_port;no_external_quantum_buffer'),
        B=0,strategy=[],certificate=dict(lower=0.,upper=0.,gap=0.,converged=True,
            arithmetic='Exact zero-key action: discard every measured record. No claim that the protocol-family optimum is zero.'))

def collect_werner(missing):
    frozen=read(ROOT/'reproduction/data/inherited/frozen_curve_delivery.json')['points'];records=[]
    for p in frozen:
        if p['model']!='Werner':continue
        q=str(p['Q']);receipt=HERE/'werner'/q/'receipt.json'
        if p['status']=='certified_optimum':
            files=list((OLD/'runs/werner'/q).glob('*/result.json'));assert len(files)==1
            path=files[0]
        elif receipt.exists() and read(receipt).get('status')=='certified':
            r=read(receipt);path=Path(r['output']);assert sha(path)==r['output_sha256']
        else:missing.append('Werner '+q);continue
        d=read(path);assert d['status']=='certified' and float(d['Q'])==float(q)
        for m,row in d['results'].items():
            assert row['lower']<=row['upper'] and row['converged']
            assert row['gap']<=row['target']
            records.append(dict(Q=q,memory=int(m[1:]),**{k:row[k] for k in ('lower','upper','gap','target')},
                source=relative(path),source_sha256=sha(path)))
    return records

def block_components():
    paths={
        '.17':OLD/'full_bb84/block_upper/0.17/edd965e00cd8bfcaeb639b2db54a4d595d2c434ca1eb63e29f46e2f9a74da718/result.json',
        '.23':OLD/'full_bb84/block_high_noise_refined/0.23/bf30471f074b11fcd8be7d47c0af86fbe8591e5a116e0be949025bf16637eb03/result.json'}
    for q in ('.19','.2','.21','.22','.23'):
        p=HERE/'bb84'/str(float(q))/'block_component/result.json'
        if p.exists():paths[q]=p
    out={}
    for q,path in paths.items():
        d=read(path);assert d['complete']
        # The Q=.23 source covers memory <=5. This is the entire N<=5
        # block component: loading all N inputs never requires >5 slots.
        assert q!='.23' or d.get('memory_cap',5)>=5
        out[str(float(q))]=dict(upper={key:value['upper'] for key,value in d['results'].items()},
            source=relative(path),sha256=sha(path),scope='Complete full-Clifford N<=5 flagged-block component; full-input attempt cost.')
    return out

def collect_bb84(missing):
    expected={'full':['.17','.19','.2','.21','.22','.23'],
              'real_xyz':['.17','.2','.23'],'real_xz':['.17','.2','.23']}
    blocks=block_components();records=[];policies={}
    for label,qq in expected.items():
        for q0 in qq:
            q=str(float(q0));root=HERE/'bb84'/q;p=root/f'refined_{label}_latest.json'
            if label=='full' and q=='0.17':p=root/'upper_latest.json'
            if not p.exists():missing.append('BB84 '+label+' '+q);continue
            path,doc=pointer(p);assert doc['complete']
            feasible_path=path.parent/'feasible.json'
            if not feasible_path.exists():feasible_path=root/'feasible.json'
            feasible=read(feasible_path)
            for memory in range(1,9):
                key=f'M{memory}';r=doc['results'][key];policy=feasible['results'][key]
                lower=float(r['lower']);upper=float(r['upper'])
                if 'ED' not in policy:
                    assert lower==0. and policy['lower']==0.,(label,q,memory,policy)
                    policy=discard_policy()
                if label=='full' and q=='0.17' and memory>=2:
                    independent=HERE/'diagnostics/nonreal_witness/result.json';d=read(independent)
                    lower=max(lower,d['certificate']['lower'])
                    policy=dict(lower=lower,ED=d['ED'],B=0,strategy=d['strategy'],certificate=d['certificate'],independent_source=relative(independent))
                independent=HERE/'bb84/0.23/symmetric_lower/cover_lower_latest.json'
                if q=='0.23' and memory==1 and label in ('full','real_xyz') and independent.exists():
                    source,d=pointer(independent)
                    if d['lower']>lower:lower=d['lower'];policy=dict(d,independent_source=relative(source))
                assert lower<=upper and policy['ED']['memory_upper_bound']<=memory
                assert policy['lower']<=max(0.,policy['certificate']['lower'])+1e-18
                tolerance=float(r.get('tolerance',max(2e-9,lower*1e-4)))
                closed=upper<=math.nextafter(lower+tolerance,math.inf)
                block=blocks.get(q) if label!='real_xz' else None
                union_upper=max(upper,block['upper'][key]) if block else None
                rid=f'{label}_Q{q}_M{memory}';policies[rid]=policy
                records.append(dict(Q=q,label=label,memory=memory,lower=lower,upper=upper,
                    tolerance=tolerance,closed=closed,union_upper=union_upper,
                    union_closed=bool(union_upper is not None and union_upper<=math.nextafter(lower+tolerance,math.inf)),
                    block_upper=block['upper'][key] if block else None,
                    N=policy['ED'].get('successful_ancestry_inputs'),
                    B=policy.get('B',policy.get('classical_B_rounds',0)),axis=policy['ED']['final_axis'],
                    policy_id=rid,source=relative(path),source_sha256=sha(path),
                    feasible_source=relative(feasible_path),feasible_sha256=sha(feasible_path)))
    return records,policies,blocks

def collect_curves(missing):
    root=HERE/'curves_augmented'
    extended=HERE/'curves_extended/status.json'
    if extended.exists() and read(extended).get('stage')=='complete':root=extended.parent
    final=HERE/'curves_final/status.json'
    if final.exists() and read(final).get('stage')=='complete':root=final.parent
    final=HERE/'curves_final/status.json'
    if final.exists() and read(final).get('stage')=='complete':root=final.parent
    policy_file=root/'policies.json'
    if not policy_file.exists():missing.append('dense portfolio');return [],{}
    policies=read(policy_file);digest=sha(policy_file);rows=[]
    for i in range(161):
        q=F(16,100)+F(i,2000);p=root/'points'/(str(q).replace('/','-')+'.json')
        if not p.exists():missing.append('dense Q '+str(q));continue
        d=read(p);assert d['policy_sha256']==digest and F(d['Q'])==q
        row=dict(Q=float(q),Q_exact=str(q),source=relative(p),source_sha256=sha(p))
        families=[name for name in ('real_xz','real_xyz','full') if name in d['lower_envelopes']]
        for family in families:
            last=0.
            for m in range(1,9):
                rate=d['lower_envelopes'][family][m];assert rate>=last;last=rate
                winner=d['winners'][family][m]
                if winner is not None:
                    assert policies[winner]['memory']<=m and family in policies[winner]['permissions']
                    result=d['results'][winner];assert result['status']=='full_interval'
                    assert rate==result['lower'] and rate<=max(0.,result['certificate']['lower'])
                row[f'{family}_R{m}']=rate;row[f'{family}_winner{m}']=winner or 'discard'
        for m in range(1,9):
            assert row['full_R'+str(m)]>=row['real_xyz_R'+str(m)]
            if 'real_xz' in families:assert row['real_xyz_R'+str(m)]>=row['real_xz_R'+str(m)]
        rows.append(row)
    return rows,policies

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    missing=[];werner=collect_werner(missing);bb84,policies,blocks=collect_bb84(missing);curves,portfolio=collect_curves(missing)
    status=dict(werner_points=len(werner)//8,expected_werner=120,bb84_points=len(bb84)//8,
        dense_points=len(curves),expected_dense=161,missing=missing,
        bb84_binary_unclosed=[r['policy_id'] for r in bb84 if not r['closed']],
        interpretation='An unclosed valid enclosure is never labelled an optimum. Union bounds require the separate small-block component.')
    out=HERE/'export';out.mkdir(exist_ok=True)
    save(out/'status.json',status);save(out/'werner_bounds.json',werner);save(out/'bb84_bounds.json',bb84)
    save(out/'bb84_policies.json',policies);save(out/'block_components.json',blocks);save(out/'dense_portfolio.json',portfolio)
    if curves:csvfile(out/'bb84_dense_curves.csv',curves)
    if args.publish:
        assert not missing,missing
        assert len(werner)==960 and len(curves)==161
        with (ROOT/'data/global_rate_landscape.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        bypoint={(r['Q'],r['memory']):r for r in werner}
        for row in rows:
            if row['source_model']!='Werner':continue
            r=bypoint[str(float(row['Q'])),int(row['memory_budget'])]
            row.update(lower=repr(r['lower']),upper=repr(r['upper']),source_status='certified_optimum',
                reproduction_status='Whole declared finite-family enclosure; optimization tolerance verified',full_protocol_record=r['source'])
        csvfile(ROOT/'data/global_rate_landscape.csv',rows)
        import shutil
        for file in out.iterdir():shutil.copyfile(file,ROOT/'data'/('revision_20261022_'+file.name))
    print(json.dumps({**status,'missing_count':len(missing),'missing':missing[:12]},indent=2))
if __name__=='__main__':main()
