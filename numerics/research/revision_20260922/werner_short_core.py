"""Re-optimize all m<=8 terminal Clifford instruments with the new objective."""
from core import *
from complete_werner_search import load
from screening import relevant_states
from ad_modes import MODES,OTPKernel,best_short
from literature_rates import watanabe
import argparse,time


def optimize(q,data,weights,offsets,agg,kernels,K=16,tolerance=2e-13):
    raw=np.concatenate([w@np.array([(1-1.5*q)**(m-j)*(q/2)**j for j in range(m+1)]) for m,w in weights.items()])
    mass=raw.sum(-1);state=raw/mass[:,None];factor=np.ones(len(state))
    best=np.maximum(0,1-entropy(state));code=np.full(len(best),0,int);mode=np.zeros(len(best),int);rounds=np.zeros(len(best),int)
    bi,si,fi=agg;maxM=max(e['M'] for e in data['entries']);incumbent=np.zeros(maxM+1)
    memories=np.array([e['M'] for e in data['entries']]);last_upper=np.zeros(len(data['entries']))
    selected=np.zeros(maxM+1,int);counts=[];otp=[OTPKernel(k.entry) for k in kernels]
    def totals():
        result=np.bincount(bi,weights=fi*mass[si]*best[si],minlength=len(data['entries']))
        for M in range(1,maxM+1):
            ids=np.flatnonzero(memories<=M);i=int(ids[result[ids].argmax()]);selected[M]=i;incumbent[M]=result[i]
        return result
    totals();active_state=np.ones(len(best),bool);start=time.time()
    for k in range(K+1):
        bound=factor*np.maximum(0,2*state.max(-1)-1)**2
        relevant,last_upper,keep=relevant_states(data,agg,mass*np.maximum(best,bound+3e-14),incumbent,padding=2e-13)
        active_state&=relevant
        ids=np.flatnonzero(active_state&(bound>best+tolerance))
        if not len(ids):break
        result=best_short(state[ids],kernels,otp);v=factor[ids]*result['rate'];take=v>best[ids];ii=ids[take]
        best[ii]=v[take];code[ii]=result['code'][take];mode[ii]=result['mode'][take];rounds[ii]=k
        v=factor[ids]*watanabe(state[ids]);take=v>best[ids];ii=ids[take]
        best[ii]=v[take];code[ii]=-2;mode[ii]=3;rounds[ii]=k
        totals();counts.append(dict(round=k,states=len(ids),instruments=int(keep.sum())))
        print('New Werner core',q,k,'states',len(ids),'seconds',round(time.time()-start,1),flush=True)
        state,s=b_step(state);factor*=s/2
    final=totals();results={}
    for M in range(1,maxM+1):
        i=int(selected[M]);e=data['entries'][i];branches=[]
        for branch in e['branches']:
            j=offsets[e['m']]+branch['state']
            if best[j]<=0:continue
            if code[j]==-2:ad=dict(kind='Watanabe_2007',mode='no_otp_hashed',n=2)
            else:ad=dict(kind='linear',code=kernels[code[j]].entry['id'],n=kernels[code[j]].n,mode=MODES[mode[j]])
            ad['classical_B_rounds']=int(rounds[j])
            branches.append(dict(**branch,AD=ad,contribution=float(branch['factor']*mass[j]*best[j]/e['m'])))
        results[f'M{M}']=dict(rate=float(final[i]),ED=e['id'],m=e['m'],M=e['M'],branches=branches,
                              final_numerical_screen_upper=float(max(final[memories<=M].max(),last_upper[memories<=M].max())))
    return dict(Q=float(q),results=results,round_counts=counts,seconds=time.time()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--q',nargs='+',type=float);args=parser.parse_args()
    data,weights,offsets,agg=load();kernels=load_codes()
    path=DATA/'werner_short_core_all_modes.json';out=json.loads(path.read_text()) if path.exists() else dict(results=[])
    done={r['Q'] for r in out['results']}
    for q in args.q or [.1,.15,.18,.2,.22,.24,.25,.26,.27,.28,.29,.3,.31,.32]:
        if q in done:continue
        result=optimize(q,data,weights,offsets,agg,kernels);out['results'].append(result)
        out['scope']='All frozen m<=8 terminal Clifford instruments and every flag; n<=7 all codes via known-state direct-sum domination, all three net Du modes plus Watanabe hashed terminal, pooled B<=16. Floating screening; independent whole-class upper certificate pending.'
        atomic_json(path,out);print('Core final',q,{k:v['rate'] for k,v in result['results'].items()},flush=True)


if __name__=='__main__':main()
