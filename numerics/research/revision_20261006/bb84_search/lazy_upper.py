"""Whole-class interval upper replay with proof-preserving lazy duals.

Every physical candidate is represented by an exact objective class. Each
class receives an outward concurrence bound, optionally tightened by a
known-state all-action bound and an exact feasible dual. A dual is omitted
only when an already rigorous bound meets the requested target. No sampled
floating rate is used to omit a candidate.
"""
from core import *
from fractions import Fraction as F
from recurrence_interval_library import generate_bounds, records, step
from recurrence_equivalence import representatives, key as canonical_key
from certify_bb84_recursive import source
from compound_menu_bounds import dual_upper
from long_code_bounds import best_bounds
from bb84_fixed import physical_kernel
import stable_intervals as iv
import hashlib
import time


def certify(path, tolerance=1e-8):
    src = json.loads(path.read_text())
    b, p, permission, nmax, K = (src[key] for key in ('b', 'p', 'permissions', 'nmax', 'K'))
    dest = path.with_name(path.stem+'_menu_certified.json')
    started = time.time()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    out = json.loads(dest.read_text()) if dest.exists() else dict(b=b, p=p, permissions=permission,
        nmax=nmax, K=K, duals={}, rounds=[])
    if out.get('source_digest') != digest or out.get('algorithm') != 'lazy_complete_interval_upper_v1':
        out['rounds'] = []
    out.update(source_digest=digest, algorithm='lazy_complete_interval_upper_v1', complete=False)
    imported = DATA/'imported_interval_dual_evidence.json'
    if imported.exists():
        evidence = json.loads(imported.read_text())
        assert evidence['b'] == b and evidence['p'] == p and evidence['nmax'] == nmax
        for key, value in evidence['duals'].items():
            if key not in out['duals'] or value['upper'] < out['duals'][key]['upper']:
                out['duals'][key] = value
        out['imported_interval_dual_evidence_sha256'] = hashlib.sha256(imported.read_bytes()).hexdigest()
    lower = np.array([0.]+[src['results'][f'M{m}']['lower'] for m in range(1,9)])
    anchors = [F(0), F(1,64), F(1,8), F(1,2), F(1)]
    pp, yy = generate_bounds(source(b,p,[(t,t) for t in anchors]), permission)
    memory, rr = records(permission)
    rep = representatives(rr, b == p)
    ids = np.flatnonzero(rep == np.arange(len(rep)))
    assert np.all(rep[rep] == rep) and np.all(memory[rep] <= memory)
    pp, yy, memory = tuple(v[ids] for v in pp), tuple(v[ids] for v in yy), memory[ids]

    def identity(ed, B):
        return json.dumps([canonical_key(ed) if b == p else ed, B], sort_keys=True)

    evaluated = {}
    for row in src['evaluated'].values():
        if 'sampled_dual' not in row:
            continue
        key = identity(row['ED'], row['classical_B_rounds'])
        if key not in evaluated or row['upper_numerical'] < evaluated[key]['upper_numerical']:
            evaluated[key] = row
    kernels = load_codes()
    physical = {kernel.entry['id']: physical_kernel(kernel) for kernel in kernels}
    for B in range(K+1):
        if B < len(out['rounds']):
            if B < K:
                pp, success = step(pp,pp,'Z')
                yy = iv.scale(iv.mul(yy,success),F(1,2))
            continue
        concurrence = np.maximum(0, iv.up(2*pp[1].max(-1)-1))
        entanglement_bound = np.minimum(yy[1], iv.up(yy[1]*iv.up(concurrence*concurrence)))
        upper, anchor = entanglement_bound.min(-1), entanglement_bound.argmin(-1)
        keys = [identity(rr[index],B) for index in ids]
        cached = 0
        for i,key in enumerate(keys):
            if key in out['duals']:
                upper[i] = min(upper[i], out['duals'][key]['upper'])
                cached += 1
        active = np.flatnonzero(upper > lower[memory]+tolerance)
        for start in range(0,len(active),128):
            local = active[start:start+128]
            a = anchor[local]
            scale = yy[1][local,a]
            target = np.maximum(0,(lower[memory[local]]+tolerance)/np.maximum(scale,1e-300))
            _, bound = best_bounds((pp[0][local,a],pp[1][local,a]),kernels,physical,nmax,target)
            upper[local] = np.minimum(upper[local], iv.up(scale*bound))
        new_duals = 0
        for i in np.flatnonzero(upper > lower[memory]+tolerance):
            key = keys[i]
            if key not in evaluated:
                continue
            candidate = dual_upper(b,p,evaluated[key],nmax)
            if key not in out['duals'] or candidate['upper'] < out['duals'][key]['upper']:
                out['duals'][key] = candidate
            upper[i] = min(upper[i], out['duals'][key]['upper'])
            new_duals += 1
            atomic_json(dest,out)
            print('BB lazy exact dual', b,p,'B',B,'node',int(ids[i]),'upper',upper[i],
                  'computed',new_duals,'seconds',round(time.time()-started,1),flush=True)
        assert np.isfinite(upper).all()
        budgets = {f'M{m}':float(upper[memory<=m].max()) for m in range(1,9)}
        unresolved = np.flatnonzero(upper > lower[memory]+tolerance)
        order = unresolved[np.argsort(-(upper[unresolved]-lower[memory[unresolved]]))]
        out['rounds'].append(dict(B=B, upper=budgets, physical_candidates=len(rr),
            exact_objective_classes=len(ids), cached_duals_used=cached, new_duals_replayed=new_duals,
            known_state_evaluations=len(active),
            unresolved=[dict(original_index=int(ids[i]),ED=rr[ids[i]],upper=float(upper[i])) for i in order]))
        atomic_json(dest,out)
        print('BB lazy full menu upper', b,p,permission,B,budgets,'unresolved',len(unresolved),
              'seconds',round(time.time()-started,1),flush=True)
        if B < K:
            pp,success = step(pp,pp,'Z')
            yy = iv.scale(iv.mul(yy,success),F(1,2))
    out['results'] = {}
    for m in range(1,9):
        key = f'M{m}'
        hi,lo = max(r['upper'][key] for r in out['rounds']),lower[m]
        assert hi+1e-14 >= lo
        out['results'][key] = dict(lower=float(lo),upper=hi,gap_upper=float(iv.up(hi-lo)),converged=hi-lo<=tolerance)
    out.update(complete=True, seconds=time.time()-started,tolerance=tolerance,
        scope='Whole registered class: every exact objective representative bounded outward; omitted dual replay means an existing rigorous bound sufficed. Physical aliases have exact source-family state/yield identities and memory dominance. No unrestricted memory-bounded optimum claim.')
    atomic_json(dest,out)

