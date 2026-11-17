"""Read-only discovery benchmark for sharper initial complete-block screening."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REVISION=HERE.parent
sys.path.insert(0,str(REVISION))
from runtime_context import configure
core,_=configure(HERE/'screening_runtime')
from bb84_search.n8_adapter import install
provider=install(core)
from core import np,atomic_json
from complete_werner_search import load
from certify_werner_joint_union import raw_bounds,replay
from screening import relevant_states
from bb84_fixed import physical_kernel
import stable_intervals as iv


def bell_relative_entropy_upper(p):
    """Outward E_R upper via monotonicity in Bell pmax, never libm entropy."""
    f=np.minimum(1,np.maximum(.5,p[1].max(-1)))
    # f is an exact binary64 upper endpoint. Its complement needs enclosure.
    complement=iv.sub((np.ones_like(f),np.ones_like(f)),(f,f))
    complement=(np.maximum(0,complement[0]),np.minimum(1,complement[1]))
    ha=iv.entropy_interval((f,f));hb=iv.entropy_interval(complement)
    lower_entropy=iv.add(ha,hb)[0]
    return np.where(p[1].max(-1)<=.5,0.,np.minimum(1,np.maximum(0,iv.up(1-lower_entropy))))


def main():
    started=time.time();data,weights,offsets,agg=load();kernels=provider.load_codes();kd={k.entry['id']:k for k in kernels}
    physical={k.entry['id']:physical_kernel(k) for k in kernels};results=[]
    for q in (.28,.30):
        runs=[]
        for manifest in (REVISION/'werner_search/runs').glob('*/manifest.json'):
            doc=json.loads(manifest.read_text());run=manifest.parent
            if float(doc['spec']['Q'])==q and (run/'data/werner_complete_joint_n128.json').exists():runs.append(run)
        run=max(runs,key=lambda p:(p/'data/werner_complete_joint_n128.json').stat().st_mtime)
        raw=raw_bounds(q,weights);mass=iv.positive_sum(raw);p=iv.div(raw,(mass[0][:,None],mass[1][:,None]))
        lower=np.zeros(9);source_snapshots=[]
        for name in ('werner_v2_full_certified.json','werner_complete_joint_n128.json'):
            path=run/'data'/name;encoded=path.read_bytes();doc=json.loads(encoded)
            source_snapshots.append(dict(path=str(path.relative_to(REVISION)),sha256=hashlib.sha256(encoded).hexdigest()))
            if name.startswith('werner_v2'):
                for row in doc['results'].values():
                    assert row['Q']==q;M=int(row['budget'][1:]);lower[M:]=np.maximum(lower[M:],row['lower'])
            else:
                for item in doc['results']:
                    assert item['Q']==q
                    for row in item['results'].values():
                        out=replay(row,raw,mass,offsets,kd,physical);lower[out['M']:]=np.maximum(lower[out['M']:],out['lower'])
        c=np.minimum(1,np.maximum(0,iv.up(2*p[1].max(-1)-1)))
        upper_c=iv.up(c*c);upper_er=bell_relative_entropy_upper(p)
        modes={}
        for name,bound in [('concurrence_squared',upper_c),('relative_entropy',upper_er)]:
            # Every flag contributes its own mass and the library's factor/m.
            flags,totals,instruments=relevant_states(data,agg,iv.up(mass[1]*bound),lower+2.5e-12)
            modes[name]=dict(active_conditional_states=int(flags.sum()),competitive_instruments=int(instruments.sum()),
                by_instrument_memory={str(M):int(sum(bool(instruments[i]) and e['M']==M for i,e in enumerate(data['entries']))) for M in range(1,5)})
        results.append(dict(Q=q,sources=source_snapshots,certified_union_incumbents=lower.tolist(),total_conditional_states=len(p[0]),
          total_instruments=len(data['entries']),initial_screen=modes,
          removed_additional_conditional_states=modes['concurrence_squared']['active_conditional_states']-modes['relative_entropy']['active_conditional_states']))
        print(json.dumps(results[-1]),flush=True)
    result=dict(status='completed',scope='Initial B=0 full-instrument screening only; all flags included, certified selected incumbents replayed. No production search was changed.',results=results,seconds=time.time()-started)
    atomic_json(HERE/'screening_benchmark.json',result)


if __name__=='__main__':main()
