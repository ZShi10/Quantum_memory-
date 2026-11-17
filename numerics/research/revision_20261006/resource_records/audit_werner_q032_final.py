"""Independent inexpensive final-record audit; no optimizer is rerun."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, math, os, sys, datetime
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REV=HERE.parent
PROJECT=REV.parent.parent
RUN=REV/'werner_search/runs/9b2cdf39cb8e0aa6f2ea'
OUT=HERE/'werner_q032_final_audit'
OUT.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=load(RUN/'manifest.json');r=load(RUN/'result.json')
c=load(RUN/'data/complete_core_upper.json')
g=load(RUN/'data/registered_upper.json')
w=load(RUN/'data/werner_v2_full_certified.json')
source_hashes=m['provenance']['source_sha256']
assert all((PROJECT/p).exists() and sha(PROJECT/p)==h for p,h in source_hashes.items())
digest=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
assert digest==m['provenance']['digest']==r['spec']['source_digest']
assert m['spec']==r['spec']
assert hashlib.sha256(json.dumps(r['spec'],sort_keys=True).encode()).hexdigest()[:20]==r['job_id']==RUN.name
assert r['status']=='certified' and r['Q']==.32
assert g['candidates']==143790 and g['evaluated_objective_representatives']==25886
assert [h['B'] for h in g['history']]==list(range(14))
remaining=g['evaluated_objective_representatives']
for h in g['history']:
    remaining-=h['removed_forever']
    assert remaining==h['states']
assert remaining==0

sys.path.insert(0,str(REV))
from runtime_context import configure
core,_=configure(OUT)
from bb84_search.n8_adapter import install
provider=install(core)
kernels=provider.load_codes()
from core import np
from bb84_fixed import physical_kernel
from certify_werner_joint_union import terminal
from resource_model.bounds import prepare_intervals
from resource_model.model import tree_from_dict,evaluate,structural_record
import stable_intervals as iv
kd={k.entry['id']:k for k in kernels}
physical={k.entry['id']:physical_kernel(k) for k in kernels}
raw=[iv.rational(v) for v in (F(13,25),F(4,25),F(4,25),F(4,25))]
source=tuple(np.array([[v[i] for v in raw]]) for i in (0,1))
replays=[];cache={}
for row in w['results'].values():
    ed=row['ED'];ad=row['AD']
    tree=tree_from_dict(ed['tree'])
    rec=structural_record(tree,ed['final_axis'],ed['permissions'])
    assert rec['memory_upper_bound']==ed['memory_upper_bound']<=int(row['budget'][1:])
    assert rec['successful_ancestry_inputs']==ed['successful_ancestry_inputs']
    assert ed['memory_is_achievable_upper_bound'] and not ed['global_minimality_proved']
    accounting=evaluate(tree,np.array([.52,.16,.16,.16]))
    assert math.isclose(accounting['expected_raw_cost'],row['expected_raw_pairs_per_ED_output'],rel_tol=2e-14)
    key=json.dumps([ed['tree'],ed['final_axis'],ad],sort_keys=True)
    if key not in cache:
        p,y=prepare_intervals(source,ed,ad['classical_B_rounds'])
        value=iv.mul(y,terminal(p,ad,kd,physical))
        cache[key]=(max(0.,float(value[0][0])),max(0.,float(value[1][0])))
    lo,hi=cache[key]
    assert (lo,hi)==(row['lower'],row['upper'])
    replays.append(dict(budget=row['budget'],tree_id=ed['tree_id'],
        memory_upper_bound=ed['memory_upper_bound'],successful_ancestry_inputs=ed['successful_ancestry_inputs'],
        final_axis=ed['final_axis'],legacy_construction=ed.get('legacy_record'),
        expected_raw_cost=row['expected_raw_pairs_per_ED_output'],AD=ad,
        lower_replayed=lo,upper_replayed=hi,
        code_parameters=kd[ad['code']].entry if ad['kind']=='linear' else None))

checks=[]
for M in range(1,9):
    key=f'M{M}';row=r['results'][key]
    lower=max([0.]+[x['lower'] for x in w['results'].values() if x['ED']['memory_upper_bound']<=M]
        +[x['lower'] for x in c['core_witnesses'] if x['M']<=M])
    upper=max(c['results'][key]['core_upper'],g['results'][key]['upper'],lower)
    gap=float(iv.up(upper-lower));target=max(r['spec']['abs_tol'],r['spec']['rel_tol']*lower)
    assert lower==row['lower'] and upper==row['upper']
    assert gap==row['gap'] and target==row['target']
    assert gap<=target and row['converged']
    checks.append(dict(budget=key,lower=lower,upper=upper,gap=gap,target=target,
        upper_controller='core' if c['results'][key]['core_upper']==upper else 'registered',
        ratio_upper_lower=(upper/lower if lower else None)))
evidence=[RUN/'manifest.json',RUN/'result.json',RUN/'data/complete_core_upper.json',
 RUN/'data/registered_upper.json',RUN/'data/werner_v2_full_certified.json',
 RUN/'data/registered_discovery.json',RUN/'data/werner_complete_joint_n128.json']
audit=dict(status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_run=str(RUN),audit_source_sha256=sha(Path(__file__)),
    source_files_verified=len(source_hashes),source_digest=digest,
    evidence_sha256={str(p.relative_to(RUN)).replace(chr(92),'/'):sha(p) for p in evidence},
    final_results_recombined=checks,independent_lower_replays=replays,
    unique_lower_replay_count=len(cache),short_code_kernels=len(kernels),
    registered_original_candidates=g['candidates'],registered_exact_representatives=g['evaluated_objective_representatives'],
    registered_last_B=13,registered_active_remaining=remaining,
    registered_suffix_explanation='At B=13 every remaining representative was removed by its retained outward source upper, which also bounds all later B prefixes. B=14..16 need no further terminal evaluation.',
    expensive_search_rerun=False,frozen_sources_modified=False,
    limitations='Certifies the declared finite achievable-bound optimization with enlarged terminal axes, not all protocols at fixed memory. M1 lower zero and positive upper do not prove zero capacity. M2..4 closure is absolute: the gap is about 12.3 percent of its tiny lower rate, not a 1e-5 relative certificate.')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:audit[k] for k in ('status','source_files_verified','unique_lower_replay_count',
 'registered_original_candidates','registered_exact_representatives','final_results_recombined')},indent=2))
