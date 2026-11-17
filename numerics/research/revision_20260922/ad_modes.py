"""Du Eq.29/30/40 kernels, with explicit net OTP costs and fixed flags.

No numerical result here is an interval certificate.  Components are signed
so a robust policy can keep the SAME flags at every unknown input value.
"""
from core import *
from functools import lru_cache
from math import comb
from streaming import deficit

MODES = ('no_otp', 'otp_plain', 'otp_compressed')


class OTPKernel:
    def __init__(self, entry):
        self.entry = entry
        self.n, self.k = entry['n'], entry['k']
        words = span(entry['rows'])
        cosets = {}
        for x in range(1 << self.n):
            cosets.setdefault(min(x ^ w for w in words), []).append(x)
        self.representatives = list(cosets)
        self.counts = np.zeros((len(cosets), self.n+1))
        for j, xs in enumerate(cosets.values()):
            for x in xs:
                self.counts[j, x.bit_count()] += 1

    def components(self, p):
        p = np.asarray(p)
        a,b,c0,c1 = bell_parameters(p)
        w = np.arange(self.n+1)
        words = a[...,None]**(self.n-w)*b[...,None]**w
        phase = (self.n-w)*binary_entropy((1-c0[...,None])/2)+w*binary_entropy((1-c1[...,None])/2)
        weighted_entropy = -xlog2x(words)+words*phase
        ps = words @ self.counts.T
        hmass = weighted_entropy @ self.counts.T+xlog2x(ps)
        gains = (self.n*ps-hmass)/self.n
        return gains, -xlog2x(ps).sum(-1)/self.n

    def evaluate(self, p):
        gains, compression = self.components(p)
        gross = np.maximum(gains,0).sum(-1)
        return np.stack((np.maximum(0,gross-(self.n-self.k)/self.n),
                         np.maximum(0,gross-compression)), -1)


def best_short(p, kernels, otp_kernels=None, modes=MODES, batch=1024):
    """Exact registered candidate iteration; only arithmetic is floating."""
    p = np.asarray(p).reshape(-1,4)
    if otp_kernels is None:
        otp_kernels = [OTPKernel(k.entry) for k in kernels]
    values = np.zeros(len(p))
    code = np.full(len(p), -1, dtype=int)
    mode = np.full(len(p), -1, dtype=int)
    # Retain per-mode optima for attribution; no tolerance-based deletion.
    by_mode = np.zeros((len(p),len(MODES)))
    for start in range(0,len(p),batch):
        stop = min(len(p),start+batch)
        pp = p[start:stop]
        powers = {n:monomials(pp,compositions(n)) for n in sorted({k.n for k in kernels})}
        for i,(kernel,otp) in enumerate(zip(kernels,otp_kernels)):
            rr = np.column_stack((kernel.evaluate_powers(powers[kernel.n]),otp.evaluate(pp)))
            for m,name in enumerate(MODES):
                if name not in modes:
                    continue
                by_mode[start:stop,m] = np.maximum(by_mode[start:stop,m],rr[:,m])
                take = rr[:,m] > values[start:stop]
                ids = start+np.flatnonzero(take)
                values[ids] = rr[take,m]
                code[ids],mode[ids] = i,m
    return dict(rate=values,code=code,mode=mode,by_mode=by_mode)


@lru_cache(None)
def rep_counts(n):
    w = np.arange(n//2+1)
    multiplicity = np.array([comb(n,int(j))/(2 if 2*j==n else 1) for j in w],dtype=float)
    return w,multiplicity


def repetition_components(p,n):
    """All relative syndromes of [n,1,n], compressed by bit weight."""
    p=np.asarray(p);a,b,c0,c1=bell_parameters(p)
    w,mult=rep_counts(n)
    u=a[...,None]**(n-w)*b[...,None]**w
    v=a[...,None]**w*b[...,None]**(n-w)
    ps=u+v
    error=np.divide(v,ps,out=np.zeros_like(v),where=ps>0)
    ca=c0[...,None]**(n-w)*c1[...,None]**w
    cb=c0[...,None]**w*c1[...,None]**(n-w)
    no=(u*deficit(ca)+v*deficit(cb)-ps*binary_entropy(error))*mult/n
    h0=binary_entropy((1-c0[...,None])/2)
    h1=binary_entropy((1-c1[...,None])/2)
    physical=-xlog2x(u)-xlog2x(v)+u*((n-w)*h0+w*h1)+v*(w*h0+(n-w)*h1)+xlog2x(ps)
    otp=(n*ps-physical)*mult/n
    cost=-(xlog2x(ps)*mult).sum(-1)/n
    return no,otp,cost


def repetition_rates(p,n):
    no,otp,cost=repetition_components(p,n)
    gross=np.maximum(otp,0).sum(-1)
    return np.stack((np.maximum(no,0).sum(-1),
                     np.maximum(gross-(n-1)/n,0),np.maximum(gross-cost,0)),-1)


def spc_otp_components(p,n):
    """Two physical-entropy components, without an exponential coset table."""
    p=np.asarray(p);h=entropy(p);d=1-2*(p[...,1]+p[...,2])
    hp=(-xlog2x(p)*np.array([1,-1,-1,1])).sum(-1)
    signs=np.array([1,-1])
    ps=(1+signs*d[...,None]**n)/2
    # Sum of -P(error) log P(error), separated by bit parity.
    raw_h=n/2*(h[...,None]+signs*d[...,None]**(n-1)*hp[...,None])
    gains=(n*ps-raw_h-xlog2x(ps))/n
    return gains,-xlog2x(ps).sum(-1)/n


def spc_otp_rates(p,n):
    gains,cost=spc_otp_components(p,n)
    gross=np.maximum(gains,0).sum(-1)
    return np.stack((np.maximum(0,gross-1/n),np.maximum(0,gross-cost)),-1)
