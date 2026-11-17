"""Registered known-input AD maximization, with recorded numerical screening.

All three modes use net secret bits. Screening uses the relative-entropy
upper bound on distillable key, after subtracting the paid OTP resource.
These are candidate searches; rigorous outward certification is separate.
"""
from core import *
from ad_modes import *
from spc_dynamic_fast import all_rates as spc_all_rates


def canonical_classical_states(p):
    """Entropy-equivalent relabelings within/between bit-error cosets.

    The error map is x -> x+c, z -> z+a*x+b. It bijectively relabels
    each code's joint (X,GZ) distribution and its public cosets. This is
    for known-input optimized flags only, never for a hidden-state policy.
    """
    a=np.sort(p[:,[0,3]],axis=-1)[:,::-1]
    b=np.sort(p[:,[1,2]],axis=-1)[:,::-1]
    swap=(b[:,0]>a[:,0])|((b[:,0]==a[:,0])&(b[:,1]>a[:,1]))
    high=np.where(swap[:,None],b,a);low=np.where(swap[:,None],a,b)
    return np.column_stack((high[:,0],low[:,0],low[:,1],high[:,1]))


def spc_upper(p,ns):
    d=1-2*(p[:,1]+p[:,2]);h=entropy(p)
    hp=(-xlog2x(p)*np.array([1,-1,-1,1])).sum(1)
    out=np.zeros((len(p),len(ns)))
    o0=2*np.sqrt(p[:,0]*p[:,3]);o1=2*np.sqrt(p[:,1]*p[:,2])
    for sign in (1,-1):
        den=1+sign*d[:,None]**ns[None,:];safe=np.maximum(den,1e-300)
        conditional=(h[:,None]+sign*d[:,None]**(ns[None,:]-1)*hp[:,None])/safe+np.log2(safe/2)/ns[None,:]
        overlap=((o0+o1)[:,None]**ns[None,:]+sign*(o0-o1)[:,None]**ns[None,:])/2
        ub=den/2*(1-1/ns[None,:]-conditional)+np.minimum(den/2,np.maximum(0,overlap))/ns[None,:]
        out+=np.where(den>1e-16,np.maximum(0,ub),0.)
    return out+3e-13


def optimize(p,y,M,records,kernels,K=16,long_max=128,tolerance=1e-13,with_watanabe=False,seeds=None):
    p=np.asarray(p).copy();y=np.asarray(y).copy();M=np.asarray(M)
    best=y*np.maximum(0,1-entropy(p))
    choices=[dict(kind='linear',code='XC0000',n=1,mode='no_otp',classical_B_rounds=0) if v>0 else None for v in best]
    by_mode=np.repeat(best[:,None],3,axis=1);counts=[];maximum=int(M.max())
    if with_watanabe:
        from literature_rates import watanabe
        value=y*watanabe(p);take=value>best
        for i in np.flatnonzero(take):
            choices[i]=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed',classical_B_rounds=0)
        best=np.maximum(best,value)
    if seeds:
        from compound_terminals import values as action_values
        for seed in seeds:
            i=seed['index'];ad=seed['AD']
            if ad is None:continue
            state=p[i:i+1].copy();scale=float(y[i])
            for _ in range(ad['classical_B_rounds']):state,s=b_step(state);scale*=float(s[0])/2
            value=scale*float(action_values(state,[ad])[0,0])
            if value>best[i]:best[i]=value;choices[i]=ad.copy()
            if ad['mode'] in MODES:by_mode[i,MODES.index(ad['mode'])]=max(by_mode[i,MODES.index(ad['mode'])],value)
    otp=[OTPKernel(k.entry) for k in kernels]
    ns=np.arange(8,long_max+1)
    incumbents=np.zeros(maximum+1)
    for budget in range(1,maximum+1):
        eligible=M<=budget
        if eligible.any():incumbents[budget]=float(best[eligible].max())
    discarded_ub=np.zeros(maximum+1)

    def update(ids,values,choice,modes):
        nonlocal best,choices
        values=np.asarray(values)
        take=values>best[ids]
        for local in np.flatnonzero(take):
            i=int(ids[local]);best[i]=float(values[local]);choices[i]=choice(local)
        for mode in range(3):
            by_mode[ids,mode]=np.maximum(by_mode[ids,mode],modes[:,mode])
        for budget in range(1,maximum+1):
            eligible=M<=budget
            if eligible.any():incumbents[budget]=max(incumbents[budget],float(best[eligible].max()))

    for rounds in range(K+1):
        upper=y*np.maximum(0,2*p.max(-1)-1)**2
        relevant=upper+3e-14>=incumbents[M]
        active=relevant&(upper+3e-14>best+tolerance)
        ids=np.flatnonzero(active)
        for budget in range(1,maximum+1):
            removed=(M<=budget)&~active
            if removed.any():discarded_ub[budget]=max(discarded_ub[budget],float(upper[removed].max()))
        if not len(ids):break
        state=p[ids];scale=y[ids]
        unique,inverse=np.unique(canonical_classical_states(state),axis=0,return_inverse=True)
        compact=best_short(unique,kernels,otp)
        r={key:value[inverse] for key,value in compact.items()}
        def short_choice(j):
            ci=int(r['code'][j]);mi=int(r['mode'][j])
            return dict(kind='linear',code=kernels[ci].entry['id'],n=kernels[ci].n,
                        mode=MODES[mi],classical_B_rounds=rounds)
        update(ids,scale*r['rate'],short_choice,scale[:,None]*r['by_mode'])
        if with_watanabe:
            value=scale*watanabe(unique)[inverse]
            update(ids,value,lambda j:dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed',classical_B_rounds=rounds),np.zeros((len(ids),3)))
        # All repetition syndrome groups, not only the zero syndrome.
        for n in ns:
            rr=repetition_rates(unique,int(n))[inverse]*scale[:,None]
            mode=rr.argmax(-1);value=rr[np.arange(len(ids)),mode]
            update(ids,value,lambda j:dict(kind='repetition',n=int(n),mode=MODES[int(mode[j])],classical_B_rounds=rounds),rr)
            oo=spc_otp_rates(unique,int(n))[inverse]*scale[:,None]
            mo=oo.argmax(-1);value=oo[np.arange(len(ids)),mo]
            update(ids,value,lambda j:dict(kind='SPC',n=int(n),mode=MODES[int(mo[j])+1],classical_B_rounds=rounds),np.column_stack((np.zeros(len(ids)),oo)))
        spc_evaluations=0
        if len(ns):
            required=np.full(len(unique),np.inf)
            # A candidate must improve the best implementable protocol at
            # its own register budget, not merely its own weaker terminal.
            # Larger-budget incumbents are monotone, so this also safely
            # screens it for every budget to which it contributes.
            objective=np.maximum(best[ids],incumbents[M[ids]])
            np.minimum.at(required,inverse,(objective+tolerance)/np.maximum(scale,1e-300))
            needed=spc_upper(unique,ns)>required[:,None]
            spc_evaluations=int(needed.sum())
            rr=spc_all_rates(unique,ns,needed)[inverse]*scale[:,None]
            ni=rr.argmax(-1);value=rr[np.arange(len(ids)),ni]
            update(ids,value,lambda j:dict(kind='SPC',n=int(ns[ni[j]]),mode='no_otp',classical_B_rounds=rounds),np.column_stack((value,np.zeros((len(ids),2)))))
        counts.append(dict(B_round=rounds,states=len(ids),distinct_AD_inputs=len(unique),SPC_evaluations=spc_evaluations))
        p,s=b_step(p);y*=s/2
    result={}
    for budget in range(1,maximum+1):
        eligible=np.flatnonzero(M<=budget)
        if not len(eligible):continue
        i=int(eligible[np.argmax(best[eligible])])
        result[f'M{budget}']=dict(rate=float(best[i]),ED=records[i],AD=choices[i],
                                  source_index=i,mode_achievable_maxima=by_mode[eligible].max(0).tolist(),
                                  largest_screened_bound=float(discarded_ub[budget]))
    return dict(results=result,round_counts=counts,
                numerical_tolerance=tolerance,
                scope='Numerical registered-class search; all modes net of OTP. No outward global-optimum certificate yet. Per-mode columns are achievable candidates, not separately optimized mode optima because screening uses the joint incumbent.')
