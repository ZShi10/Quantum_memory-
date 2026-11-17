"""Check arbitrary-code and long-family interval terminals independently."""
from compound_terminals import *
from certify_recursive import repetition_all
from certify_werner_witnesses import kernel_rate,spc_rate,xlog
from interval_certificate import IV
from prove_single_round_counterexample import ivpower,positive
from long_code_bounds import LongBounds


def otp_reference(p,a):
    n=a['n'];mode=a['mode'];x=p[0]+p[3];y=p[1]+p[2]
    h0=-(xlog(p[0])+xlog(p[3])-xlog(x))/x if x.lo>0 else IV(0)
    h1=-(xlog(p[1])+xlog(p[2])-xlog(y))/y if y.lo>0 else IV(0)
    if a['kind']=='linear':
        k=kernels()[a['code']];rows=OTPKernel(k.entry).counts;mult=[1]*len(rows);dimension=k.k
    elif a['kind']=='SPC':
        rows=[[comb(n,w) if w%2==s else 0 for w in range(n+1)] for s in (0,1)];mult=[1,1];dimension=n-1
    else:
        rows=[];mult=[];dimension=1
        for w in range(n//2+1):
            row=[0]*(n+1);row[w]+=1;row[n-w]+=1;rows.append(row);mult.append(F(comb(n,w),2 if w*2==n else 1))
    word=[ivpower(x,n-w)*ivpower(y,w) for w in range(n+1)]
    ent=[-xlog(u)+u*((n-w)*h0+w*h1) for w,u in enumerate(word)]
    gross=IV(0);cost=IV(0)
    for row,m in zip(rows,mult):
        mass=sum((int(c)*u for c,u in zip(row,word)),IV(0))
        hmass=sum((int(c)*u for c,u in zip(row,ent)),IV(0))+xlog(mass)
        gross=gross+IV(m)*positive(n*mass-hmass);cost=cost-IV(m)*xlog(mass)
    return (gross-(IV(n-dimension) if mode=='otp_plain' else cost))/IV(n)


def main():
    menu=[]
    for k in kernels().values():
        if k.n<=3 or k.n==7:
            for mode in MODES:menu.append(dict(kind='linear',code=k.entry['id'],n=k.n,mode=mode))
    for n in (8,13,24,128):
        for mode in MODES:menu.append(dict(kind='repetition',n=n,mode=mode))
    for n in (8,13):
        for mode in MODES:menu.append(dict(kind='SPC',n=n,mode=mode))
    for n in (24,128):
        for mode in MODES[1:]:menu.append(dict(kind='SPC',n=n,mode=mode))
    checks=0;vector_checks=0;max_width=0.;max_float_error=0.
    for numerators in ((79,3,7,11),(52,24,1,23),(5,19,71,5)):
        p=[F(v,100) for v in numerators];pp=[IV(v) for v in p];bounds=[iv.rational(v) for v in p]
        jet=j.Jet((np.array([[v[0] for v in bounds]]),np.array([[v[1] for v in bounds]])))
        long=LongBounds(jet.v);long_cache={}
        numeric=values(np.array([list(map(float,p))]),menu)[0]
        for i,a in enumerate(menu):
            if a['mode']!='no_otp':reference=otp_reference(pp,a)
            elif a['kind']=='linear':reference=kernel_rate(pp,kernels()[a['code']])
            elif a['kind']=='SPC':reference=spc_rate(pp,a['n'])
            else:reference=repetition_all(pp,a['n'])
            value=terminal_jet(jet,a).v;lo,hi=float(value[0][0]),float(value[1][0])
            assert lo<=float(reference.lo) and hi>=float(reference.hi),(numerators,a,lo,hi,reference.lo,reference.hi)
            if a['kind']!='linear':
                key=(a['kind'],a['n'])
                if key not in long_cache:
                    if a['kind']=='repetition':long_cache[key]=long.repetition(a['n'])
                    else:
                        upper,plain,compressed=long.spc(a['n']);exact=long.spc_exact(a['n'],np.array([0])) if a['n']<=13 else None
                        long_cache[key]=(exact,plain,compressed)
                        if a['n']<=13:
                            target=spc_rate(pp,a['n']);assert upper[1][0]>=float(target.hi);vector_checks+=1
                vector=long_cache[key][MODES.index(a['mode'])]
                if vector is not None:
                    # The vector engine is a KNOWN-input optimization bound,
                    # where skipping a whole paid mode is allowed. The jet
                    # engine above remains signed for compound policies.
                    assert vector[0][0]<=max(0.,float(reference.lo)) and vector[1][0]>=max(0.,float(reference.hi)),(a,vector,reference.lo,reference.hi)
                    vector_checks+=1
            max_width=max(max_width,hi-lo);max_float_error=max(max_float_error,abs(numeric[i]-float(reference.lo)));checks+=1
    assert max_float_error<3e-12,max_float_error
    out=dict(status='passed',checks=checks,vector_long_code_checks=vector_checks,largest_interval_width=max_width,largest_float_error=max_float_error,
             reference='Independent 60-digit exact-rational entropy and integer coset/type counts; all signs of OTP net retained; long repetition/OTP through 128 and SPC no-OTP through 13.')
    atomic_json(DATA/'compound_terminal_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
