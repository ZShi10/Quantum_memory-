"""Publish a distinctly labeled complete-block refinement certificate.

The bridge schema is compatible with existing read-only collectors, but its
algorithm and full dependency hashes explicitly identify it as a composition
of independently computed outward bounds, never an old audit_upper run.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil

HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
PROJECT=REV.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--refinement',type=Path,action='append',required=True)
    parser.add_argument('--registered',type=Path,required=True)
    parser.add_argument('--publish-bridge',action='store_true')
    args=parser.parse_args()
    base=json.loads(args.base.read_text())
    assert base['complete']
    refinements=[json.loads(path.read_text()) for path in args.refinement]
    registered=json.loads(args.registered.read_text())
    assert registered['complete'] and registered['b']==base['b'] and registered['p']==base['p']
    entries={e['ED']:dict(e) for e in base['entries']}
    assert len(entries)==len(base['entries'])==265
    sources={str(p.resolve().relative_to(PROJECT)):sha(p) for p in
             [Path(__file__),args.base,args.registered,*args.refinement]}
    sources.update(base['inputs'])
    sources.update(base['lower_sources'])
    for refined in refinements:
        assert refined['complete'] and refined['base_fingerprint']==base['job_fingerprint']
        for key in ('b','p','permissions','fixed_block_max','general_code_max','long_max','B_max'):
            assert refined[key]==base[key],key
        assert len(refined['entries'])==len(entries)
        assert {row['ED'] for row in refined['entries']}==set(entries)
        for key,value in refined['inputs'].items():
            assert key not in sources or sources[key]==value,key
            sources[key]=value
        for row in refined['entries']:
            assert row['m']==entries[row['ED']]['m'] and row['M']==entries[row['ED']]['M']
            assert math.isfinite(row['upper']) and row['upper']>=0
            entries[row['ED']]['upper']=min(entries[row['ED']]['upper'],row['upper'])
    for path,value in sources.items():assert sha(PROJECT/path)==value,path
    identity=dict(certificate_schema='qkd.complete_block_upper.refinement.v1',
                  algorithm='per_instrument_minimum_of_outward_point_test_bounds',
                  b=base['b'],p=base['p'],permissions=base['permissions'],
                  fixed_block_max=5,general_code_max=8,long_max=128,B_max=16,
                  base_fingerprint=base['job_fingerprint'],
                  refinement_fingerprints=[r['job_fingerprint'] for r in refinements],
                  inputs=sources)
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=HERE/'combined'/tag;folder.mkdir(parents=True,exist_ok=True)
    results={};union={}
    for M in range(1,9):
        memory=f'M{M}';r=registered['results'][memory]
        upper=max(e['upper'] for e in entries.values() if e['M']<=M)
        lower=r['lower'];target=registered['required_gaps'][memory]
        results[memory]=dict(upper=upper,lower_witness=lower,
            dominated_within_tolerance=upper<=lower+target)
        hi=max(upper,r['upper']);gap=math.nextafter(hi-lower,math.inf)
        union[memory]=dict(lower=lower,upper=hi,gap_upper=gap,
                          required_gap=target,converged=gap<=target)
    result=dict(**identity,job_fingerprint=tag,complete=True,
                union_gap_closed=all(row['converged'] for row in union.values()),
                entry_count=len(entries),entries=list(entries.values()),results=results,
                registered_union_results=union,
                lower_sources=base['lower_sources'],
                proof='Each retained per-instrument bound is outward and covers the full B/AD menu. Taking their minimum preserves that bound; taking the maximum over every instrument with M<=budget bounds the complete fixed-block class. Each additional test t is an allowed adversarial input. The selected protocol is not allowed to adapt to hidden t.',
                scope='Complete real m<=5 flagged one-terminal-bit Clifford instruments and original n<=8/endpoint128/Watanabe/B16 menu. This is a derived certificate, not an audit_upper.py execution. It does not change the protocol class or assert unrestricted fixed-memory optimality.')
    assert result['union_gap_closed']
    encoded=json.dumps(result,indent=2).encode()
    (folder/'manifest.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    (folder/'result.json').write_bytes(encoded)
    snapshot=folder/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,value in sources.items():
        p=PROJECT/name;target=snapshot/(value+p.suffix)
        if not target.exists():target.write_bytes(p.read_bytes())
    validation=dict(status='passed',job_fingerprint=tag,result_sha256=hashlib.sha256(encoded).hexdigest(),
                    all_265_identifiers_preserved=True,source_hashes_verified=len(sources),
                    same_model_and_menus=True,union_gap_closed=True,
                    no_new_lower_claimed=True,registered_union_results=union)
    (folder/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if args.publish_bridge:
        bridge=REV/'code_extension/bb84_m5/upper_runs'/('refinement_'+tag)
        bridge.mkdir(parents=True,exist_ok=True)
        for filename in ('result.json','manifest.json','validation.json'):
            destination=bridge/filename
            if destination.exists():assert destination.read_bytes()==(folder/filename).read_bytes()
            else:shutil.copyfile(folder/filename,destination)
        validation['bridge']=str(bridge/'result.json')
    print(json.dumps(dict(folder=str(folder),**validation),indent=2))


if __name__=='__main__':main()
