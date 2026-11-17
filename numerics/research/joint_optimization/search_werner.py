"""Joint Clifford/short-code frontier for known Werner input.

One block m<=4, fixed output measurement, separate public ED branches,
single-round n<=6 AD. The optional pooled B prefix is tabulated separately.
"""
from pathlib import Path
import json,time,csv,argparse
import numpy as np
from algebra import compositions,code_components
from rates import CodeKernel,best_codes

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'


def collapse_werner(entry):
    tensor=np.asarray(entry['tensor']);m=entry['m']
    result=np.zeros(tensor.shape[:2]+(m+1,),dtype=int)
    for j,t in enumerate(compositions(m)):result[:,:,m-t[0]]+=tensor[:,:,j]
    return result


def werner_blocks(entries):
    seen={}
    for entry in entries:
        weights=collapse_werner(entry)
        signature=weights[0].tobytes()+b''.join(sorted(row.tobytes() for row in weights[1:]))
        key=(entry['m'],entry['real'],signature)
        if key not in seen or (entry['M'],len(entry['gates']))<(seen[key]['M'],len(seen[key]['gates'])):
            seen[key]={**entry,'werner_weights':weights.tolist()}
    return list(seen.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',type=int,default=0)
    parser.add_argument('--families',action='store_true')
    args=parser.parse_args();start=time.time()
    codes=json.loads((DATA/'codes.json').read_text())['entries']
    # This removal is exact for the known-input single-code envelope: a
    # direct-sum code cannot beat the better of its component codes.
    codes=[e for e in codes if len(code_components(e['n'],e['rows']))==1]
    kernels=[CodeKernel(e) for e in codes]
    entries=werner_blocks(json.loads((DATA/'blocks.json').read_text())['entries'])
    q=np.unique(np.round(np.r_[np.arange(.01,.3001,.005),np.arange(.301,.3321,.001),.24,.25,.275,.28,.29,.30,.31,.32,.33],8))
    if args.families:
        entries+=json.loads((DATA/'graph_family_blocks.json').read_text())['entries']
        q=np.array([.10,.15,.18,.20,.22,.24,.25,.26,.27,.275,.28,.285,.29,.295,.30,.305,.31,.315,.32,.325,.33])
    f=1-1.5*q;t=q/2
    branches=[];weights=[];block_indices=[]
    for bi,e in enumerate(entries):
        m=e['m'];powers=np.stack([f**(m-w)*t**w for w in range(m+1)],-1)
        raw=np.einsum('qw,sew->qse',powers,np.array(e['werner_weights']))
        ps=raw.sum(-1);p=np.divide(raw,ps[...,None],out=np.full_like(raw,.25),where=ps[...,None]>0)
        for s in range(len(raw[0])):
            branches.append(p[:,s]);weights.append(ps[:,s]/m);block_indices.append((bi,s))
    states=np.array(branches);weight=np.array(weights)
    best=np.zeros(states.shape[:-1]);endpoint=best.copy();oneway=best.copy()
    chosen=np.full(best.shape,-1,dtype=int);rounds_chosen=np.zeros(best.shape,dtype=int)
    code_endpoint=chosen.copy();round_endpoint=rounds_chosen.copy()
    state=states.copy();factor=np.ones(best.shape)
    for k in range(args.rounds+1):
        values=best_codes(state.reshape(-1,4),kernels)
        r=values['rate'].reshape(best.shape)*factor
        re=values['endpoint_rate'].reshape(best.shape)*factor
        take=r>best+1e-14;best[take]=r[take]
        chosen[take]=values['code'].reshape(best.shape)[take];rounds_chosen[take]=k
        take=re>endpoint+1e-14;endpoint[take]=re[take]
        code_endpoint[take]=values['endpoint_code'].reshape(best.shape)[take];round_endpoint[take]=k
        if k==0:oneway=values['oneway'].reshape(best.shape)
        a,x,y,z=np.moveaxis(state,-1,0)
        out=np.stack((a*a+z*z,x*x+y*y,2*x*y,2*a*z),-1)
        s=out.sum(-1);factor*=s/2;state=out/s[...,None]
        print('joint AD prefix rounds',k,'branches',len(branches),'seconds',round(time.time()-start,1),flush=True)
    total=np.zeros((len(entries),len(q)));zero=total.copy();end=total.copy();direct=total.copy()
    branch_lookup={}
    for i,(bi,s) in enumerate(block_indices):
        total[bi]+=weight[i]*best[i];end[bi]+=weight[i]*endpoint[i];direct[bi]+=weight[i]*oneway[i]
        if s==0:zero[bi]=weight[i]*best[i]
        branch_lookup[bi,s]=i
    results={};rows=[]
    for mode in ('real','full'):
        for budget in range(1,5):
            idx=[i for i,e in enumerate(entries) if e['M']<=budget and (mode=='full' or e['real'])]
            values=total[idx];ci=np.argmax(values,axis=0);actual=np.array(idx)[ci]
            curve=values[ci,np.arange(len(q))]
            end_curve=end[idx].max(0);zero_curve=zero[idx].max(0)
            records=[]
            for j,bi in enumerate(actual):
                e=entries[bi]
                branch_codes=[]
                for s in range(1<<(e['m']-1)):
                    i=branch_lookup[bi,s];c=chosen[i,j]
                    branch_codes.append(dict(syndrome=s,code=codes[c]['id'] if c>=0 else None,
                                             binary_rounds=int(rounds_chosen[i,j]),contribution=float(weight[i,j]*best[i,j])))
                records.append(dict(Q=float(q[j]),rate=float(curve[j]),ED=e['id'],M=e['M'],m=e['m'],branches=branch_codes))
                rows.append([mode,budget,q[j],curve[j],end_curve[j],zero_curve[j]])
            results[f'{mode}_M{budget}']=records
    output=dict(scope=dict(m_max=4,n_max=6,pooled_binary_prefix_rounds=args.rounds,
                ED_branch_handling='All public branches separately pooled; one fixed logical Z for the quantum block.',
                optimality='Finite library of explicitly compiled instruments; no minimum-width or all-LOCC assertion.'),
                code_orbits=124,indecomposable_codes=len(codes),werner_instruments=len(entries),
                seconds=time.time()-start,results=results)
    suffix=f'k{args.rounds}'
    if args.families:
        suffix+='f';output['scope']['m_max']=8
        output['scope']['optimality']='Complete short m<=4 library plus explicitly constructed m=5..8 graph families. Larger blocks are not exhaustively enumerated. M is a certified implementation upper bound.'
    (DATA/f'werner_joint_{suffix}.json').write_text(json.dumps(output,indent=2))
    with (DATA/f'werner_joint_{suffix}.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['permissions','memory_budget','Q','joint_all_branches','endpoint_codes','zero_ED_branch']);writer.writerows(rows)
    print('Completed',len(entries),'Werner instruments;',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':main()
