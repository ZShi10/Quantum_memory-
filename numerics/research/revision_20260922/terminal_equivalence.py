"""Reuse certified fixed policies through exact terminal B identities.

ZZ/Z and XX/Z equal B after Z/Z and X/X input measurements; YY/Y
equals B after Y/Y up to exchanging the two Bell errors in the bit-one
sector. Every registered terminal is invariant under that exchange.
Allocation changes are bounded by their exact L1 distance, since every
signed per-input action lies in [-1,1] and quantum/classical yield <=1.
"""
from core import *
from fractions import Fraction as F
import stable_intervals as iv


def canonical(ed,B):
    if ed['family']!='real_decoder_recurrence':
        return ('pump',ed.get('primitive'),ed.get('r'),ed['final_axis'],B)
    word=ed['path'];axis=ed['final_axis']
    while word:
        pair=word[-1],axis
        if pair==('Z','Z'):axis='Z'
        elif pair==('X','Z'):axis='X'
        elif pair==('R','Y'):axis='Y'
        else:break
        word=word[:-1];B+=1
    return 'recurrence',word,axis,B


def weights(strategy):
    result={}
    for item in strategy:
        key=json.dumps(item['AD'],sort_keys=True)
        result[key]=result.get(key,F(0))+F(item['allocation_exact'])
    return result


def solved_equivalent(evaluated,ed,B):
    """Transfer an entire fixed-menu problem, including its feasible dual.

    This does not assert convergence: the original upper/lower gap is kept.
    Exact identities preserve every action, including its paid OTP cost.
    """
    import copy
    target=canonical(ed,B)
    if target[0]!='recurrence':return None
    matches=[(key,row) for key,row in evaluated.items()
             if 'sampled_dual' in row and canonical(row['ED'],row['classical_B_rounds'])==target]
    if not matches:return None
    key,row=min(matches,key=lambda pair:(pair[1]['upper_numerical']-pair[1]['lower'],-pair[1]['lower']))
    result=copy.deepcopy(row)
    result.update(ED=ed,classical_B_rounds=B,
                  exact_equivalent_problem=dict(source_evaluated_row=key,canonical=target,
                    proof='Integer polynomial terminal-B identities and actionwise Bell-label invariance; same fixed-menu primal and dual feasible points.'))
    return result


def reuse(b,p,ed,B,strategy,atol):
    target=canonical(ed,B)
    if target[0]!='recurrence':return None
    w=weights(strategy);best=None
    for path in DATA.glob(f'bb84_joint_extension_*_{b:.5f}_{p:.5f}_n128.json'):
        src=json.loads(path.read_text())
        for key,row in src.get('evaluated',{}).items():
            old=row.get('certificate')
            if not old or not row.get('strategy'):continue
            if canonical(row['ED'],row['classical_B_rounds'])!=target:continue
            ww=weights(row['strategy']);distance=sum((abs(w.get(k,F(0))-ww.get(k,F(0))) for k in w.keys()|ww.keys()),F(0))
            if distance>F(str(atol)):continue
            margin=float(iv.rational(distance)[1]);lo=float(iv.down(old['lower']-margin));hi=float(iv.up(old['upper']+margin))
            gap=float(iv.up(hi-lo))
            if gap>atol:continue
            result=dict(lower=lo,upper=hi,gap=gap,argmin=old['argmin'],cells=0,converged=True,atol=atol,
                        arithmetic='Exact polynomial terminal-B identity; invariant Bell-label swap; outward exact-L1 allocation perturbation of a saved whole-correlation certificate.',
                        equivalent_source=path.name,equivalent_row=key,allocation_L1_exact=str(distance),canonical=target)
            if best is None or gap<best['gap']:best=result
    return best


def verify():
    from recurrence_v2 import MAPS,prepare
    from recursive_library import PRE
    from itertools import combinations_with_replacement
    from compound_terminals import actions,values
    pairs=list(combinations_with_replacement(range(4),2));identities=[]
    for primitive,axis,input_axis,perm in [('Z','Z','Z',(0,1,2,3)),('X','Z','X',(0,1,2,3)),('R','Y','Y',(0,2,1,3))]:
        a=np.zeros((4,10),int);b=a.copy()
        for i,j,k in MAPS[primitive]:a[list(PRE[axis]).index(k),pairs.index(tuple(sorted((i,j))))]+=1
        for i,j,k in MAPS['Z']:b[k,pairs.index(tuple(sorted((PRE[input_axis][i],PRE[input_axis][j]))))]+=1
        assert np.array_equal(a,b[list(perm)])
        identities.append(dict(primitive=primitive,axis=axis,input_axis=input_axis,permutation=perm,coefficients=a.tolist()))
    rng=np.random.default_rng(26581);p=rng.dirichlet(np.ones(4),24)
    menu=actions(17);a=values(p,menu);b=values(p[:,[0,2,1,3]],menu)
    error=float(np.max(np.abs(a-b)));assert error<3e-12
    atomic_json(DATA/'terminal_B_equivalence_checks.json',dict(status='passed',exact_polynomial_identities=identities,
               terminal_swap_comparisons=int(a.size),max_discrepancy=error,
               proof='Swapping X/Y changes Z^n by X^n, so (X^n,GZ^n) is relabeled invertibly conditional on X^n; physical entropy and syndrome are unchanged. The Watanabe expression is invariant by binary entropy symmetry.'))
    print('Terminal identities: 3 exact; action comparisons',a.size,'max discrepancy',error,flush=True)


if __name__=='__main__':verify()
