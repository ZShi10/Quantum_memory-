"""Complete fixed m<=8 terminal instruments with the common long AD menu."""
from core import *
from complete_werner_search import load
from screening import relevant_states
from ad_modes import MODES,OTPKernel,best_short,repetition_rates,spc_otp_rates
from known_search import canonical_classical_states,spc_upper
from spc_dynamic_fast import all_rates as spc_rates
from literature_rates import watanabe
import argparse,time


def optimize(q,data,weights,offsets,agg,kernels,K=16,nmax=128,tolerance=2e-13):
    raw=np.concatenate([w@np.array([(1-1.5*q)**(m-j)*(q/2)**j for j in range(m+1)]) for m,w in weights.items()])
    mass=raw.sum(-1);state=raw/mass[:,None];factor=np.ones(len(state))
    best=np.maximum(0,1-entropy(state));code=np.zeros(len(best),int);mode=np.zeros(len(best),int)
    rounds=np.zeros(len(best),int);length=np.ones(len(best),int)
    bi,si,fi=agg;maxM=max(e['M'] for e in data['entries']);incumbent=np.zeros(maxM+1)
    memories=np.array([e['M'] for e in data['entries']]);last_upper=np.zeros(len(data['entries']))
    weight=np.zeros(len(state));np.maximum.at(weight,si,fi)
    state_tolerance=1e-11/np.maximum(weight*mass*int(np.bincount(bi).max()),1e-300)
    selected=np.zeros(maxM+1,int);counts=[];otp=[OTPKernel(k.entry) for k in kernels]
    def totals():
        result=np.bincount(bi,weights=fi*mass[si]*best[si],minlength=len(data['entries']))
        for M in range(1,maxM+1):
            ids=np.flatnonzero(memories<=M);i=int(ids[result[ids].argmax()]);selected[M]=i;incumbent[M]=result[i]
        return result
    def update(ids,v,cc,mm,nn,B):
        take=v>best[ids];ii=ids[take];best[ii]=v[take]
        code[ii]=np.broadcast_to(cc,len(ids))[take];mode[ii]=np.broadcast_to(mm,len(ids))[take]
        length[ii]=np.broadcast_to(nn,len(ids))[take];rounds[ii]=B
    # Existing short-menu witnesses are valid incumbents in this enlarged
    # class. Replay their actual branches; do not import a numerical upper.
    seed_path=DATA/'werner_short_core_all_modes.json'
    if seed_path.exists():
        from compound_terminals import values
        by_id={e['id']:e for e in data['entries']};ki={k.entry['id']:i for i,k in enumerate(kernels)}
        for row in json.loads(seed_path.read_text())['results']:
            if row['Q']!=q:continue
            for seed in row['results'].values():
                entry=by_id[seed['ED']]
                for branch in seed['branches']:
                    j=offsets[entry['m']]+branch['state'];ad=branch['AD'];pp=state[j:j+1].copy();yy=1.
                    for _ in range(ad['classical_B_rounds']):pp,s=b_step(pp);yy*=float(s[0])/2
                    value=float(values(pp,[ad])[0,0])*yy
                    cc=-2 if ad['kind']=='Watanabe_2007' else ki[ad['code']]
                    mm=3 if cc==-2 else MODES.index(ad['mode'])
                    update(np.array([j]),np.array([value]),cc,mm,ad['n'],ad['classical_B_rounds'])
    totals();active_state=np.ones(len(best),bool);start=time.time();ns=np.arange(8,nmax+1)
    for B in range(K+1):
        bound=factor*np.maximum(0,2*state.max(-1)-1)**2
        relevant,last_upper,keep=relevant_states(data,agg,mass*np.maximum(best,bound+3e-14),incumbent,padding=2e-13)
        active_state&=relevant;ids=np.flatnonzero(active_state&(bound>best+state_tolerance))
        if not len(ids):break
        unique,inverse=np.unique(canonical_classical_states(state[ids]),axis=0,return_inverse=True)
        result=best_short(unique,kernels,otp);scale=factor[ids]
        update(ids,scale*result['rate'][inverse],result['code'][inverse],result['mode'][inverse],
               np.array([kernels[i].n for i in result['code']])[inverse],B)
        update(ids,scale*watanabe(unique)[inverse],-2,3,2,B)
        totals()
        relevant,last_upper,keep=relevant_states(data,agg,mass*np.maximum(best,bound+3e-14),incumbent,padding=2e-13)
        active_state&=relevant;ids=ids[active_state[ids]&(bound[ids]>best[ids]+state_tolerance[ids])]
        if len(ids):
            unique,inverse=np.unique(canonical_classical_states(state[ids]),axis=0,return_inverse=True);scale=factor[ids]
        else:unique=np.empty((0,4));inverse=np.array([],int);scale=np.array([])
        for n in ns:
            if not len(ids):break
            rr=repetition_rates(unique,int(n))[inverse]*scale[:,None];mi=rr.argmax(-1)
            update(ids,rr[np.arange(len(ids)),mi],-3,mi,n,B)
            oo=spc_otp_rates(unique,int(n))[inverse]*scale[:,None];mi=oo.argmax(-1)
            update(ids,oo[np.arange(len(ids)),mi],-4,mi+1,n,B)
        if len(ns) and len(ids):
            threshold=np.full(len(unique),np.inf);np.minimum.at(threshold,inverse,(best[ids]+state_tolerance[ids])/np.maximum(scale,1e-300))
            needed=spc_upper(unique,ns)>threshold[:,None];rr=spc_rates(unique,ns,needed)[inverse]*scale[:,None]
            ni=rr.argmax(-1);update(ids,rr[np.arange(len(ids)),ni],-4,0,ns[ni],B)
        totals();counts.append(dict(B=B,states=len(ids),distinct_states=len(unique),instruments=int(keep.sum())))
        print('Complete joint Werner',q,B,'states',len(ids),'distinct',len(unique),'seconds',round(time.time()-start,1),flush=True)
        state,s=b_step(state);factor*=s/2
    final=totals();results={}
    for M in range(1,maxM+1):
        i=int(selected[M]);e=data['entries'][i];branches=[]
        for branch in e['branches']:
            j=offsets[e['m']]+branch['state']
            if best[j]<=0:continue
            ad=dict(kind={-2:'Watanabe_2007',-3:'repetition',-4:'SPC'}.get(code[j],'linear'),
                    mode='no_otp_hashed' if mode[j]==3 else MODES[mode[j]],n=int(length[j]),classical_B_rounds=int(rounds[j]))
            if code[j]>=0:ad['code']=kernels[code[j]].entry['id']
            branches.append(dict(**branch,AD=ad,contribution=float(branch['factor']*mass[j]*best[j]/e['m'])))
        results[f'M{M}']=dict(rate=float(final[i]),ED=e['id'],m=e['m'],M=e['M'],branches=branches,
                              final_numerical_screen_upper=float(max(final[memories<=M].max(),last_upper[memories<=M].max())))
    return dict(Q=float(q),results=results,round_counts=counts,seconds=time.time()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float);parser.add_argument('--n',type=int,default=128)
    a=parser.parse_args();data,weights,offsets,agg=load();kernels=load_codes();path=DATA/f'werner_complete_joint_n{a.n}.json'
    out=json.loads(path.read_text()) if path.exists() else dict(results=[]);done={r['Q'] for r in out['results']}
    for q in a.q or [.1,.2,.25,.28,.3,.32]:
        if q in done:continue
        result=optimize(q,data,weights,offsets,agg,kernels,nmax=a.n);out['results'].append(result)
        out['scope']='Every frozen fixed Clifford terminal instrument m<=8, every ED flag with its own B<=16 and classical action; all n<=7 code classes via direct sums, repetition/SPC<=nmax, three paid Du modes and Watanabe. Floating candidate discovery; whole-class interval certificate separate.'
        atomic_json(path,out);print('Complete joint final',q,{k:v['rate'] for k,v in result['results'].items()},flush=True)


if __name__=='__main__':main()
