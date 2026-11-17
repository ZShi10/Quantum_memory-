"""Outward whole-menu bound for every registered recursive/tree candidate."""
from fractions import Fraction as F
import time
import json
from core import np, atomic_json
import stable_intervals as iv
from recurrence_interval_library import generate_bounds as old_bounds, records as old_records, step
from resource_model.bounds import generate_bounds as tree_bounds
from resource_model.catalog import candidate_records
from long_code_bounds import best_bounds
from resource_records.werner_quotient import restrict_full_arrays


def certify(q_string,kernels,physical,lower,tolerance,run):
    start=time.time()
    q=F(q_string)
    row=[iv.rational(v) for v in (1-F(3,2)*q,q/2,q/2,q/2)]
    initial=tuple(np.array([[v[i] for v in row]]) for i in (0,1))
    p,y=old_bounds(initial,'full',7,160)
    M,_=old_records('full',7,160)
    ep,ey=tree_bounds(initial,'full',5,7)
    em=np.array([r['memory_upper_bound'] for _,r in candidate_records(5,7,'full')])
    p=tuple(np.concatenate((p[j][:,0,:],ep[j][:,0,:])) for j in (0,1))
    y=tuple(np.concatenate((y[j][:,0],ey[j][:,0])) for j in (0,1))
    M=np.concatenate((M,em))
    assert len(M)==67455+76335
    p,y,M,_,quotient_proof=restrict_full_arrays(p,y,M)
    del ep,ey,em
    upper=np.zeros(9);active=np.ones(len(M),bool);history=[]
    checkpoint=run/'data/registered_upper_checkpoint.json'
    # Checkpoints are progress records. Restarting regenerates every state,
    # so no stale active-index array is ever implicitly reused.
    for B in range(17):
        concurrence=np.maximum(0,iv.up(2*p[1].max(-1)-1))
        er=np.minimum(y[1],iv.up(y[1]*iv.up(concurrence*concurrence)))
        removed=active&(er<=lower[M]+tolerance)
        active[removed]=False
        for budget in range(1,9):
            ids=np.flatnonzero(removed&(M<=budget))
            if len(ids):upper[budget]=max(upper[budget],float(er[ids].max()))
        ids=np.flatnonzero(active)
        if not len(ids):
            history.append(dict(B=B,states=0,removed_forever=int(removed.sum())))
            break
        target=np.maximum(0,(lower[M[ids]]+tolerance)/np.maximum(y[1][ids],1e-300))
        lo,hi=best_bounds((p[0][ids],p[1][ids]),kernels,physical,128,target)
        bound=np.minimum(er[ids],iv.up(y[1][ids]*hi))
        assert np.isfinite(bound).all()
        for budget in range(1,9):
            selected=M[ids]<=budget
            if selected.any():upper[budget]=max(upper[budget],float(bound[selected].max()))
        history.append(dict(B=B,states=len(ids),removed_forever=int(removed.sum()),
                            largest_excess=float(np.max(bound-lower[M[ids]]))))
        atomic_json(checkpoint,dict(Q=q_string,history=history,upper=upper.tolist(),seconds=time.time()-start))
        print('REGISTERED_UPPER',q_string,B,len(ids),round(time.time()-start,2),flush=True)
        if B<16:
            state,success=step((p[0][ids],p[1][ids]),(p[0][ids],p[1][ids]),'Z')
            scale=iv.scale(iv.mul((y[0][ids],y[1][ids]),success),F(1,2))
            for j in (0,1):p[j][ids]=state[j];y[j][ids]=scale[j]
    results={f'M{m}':dict(upper=float(upper[m]),lower_incumbent=float(lower[m])) for m in range(1,9)}
    return dict(results=results,history=history,seconds=time.time()-start,candidates=143790,
                evaluated_objective_representatives=len(M),exact_quotient=quotient_proof,
                scope='All 67455 original four-decoder homogeneous/constant-pumping records and all 76335 extension records. Every allowed B<=16 and complete n<=8 short-code/three-mode menu, rep/SPC<=128, Watanabe. Outward value bounds; lower incumbents may come from either registered or complete-block witnesses.')
