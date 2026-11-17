"""Revision-local probability engine with stable entropy extrema.

Reuse the audited elementary/log arithmetic without modifying frozen code.
For h(x)=-x log2 x, concavity puts its minimum at a box endpoint and its
maximum at an endpoint or 1/e. Two endpoint logs avoid a mean-value slope
overflow when one probability is subnormal and the other is merely small.
"""
from float_enclosures import *


def entropy_interval(x):
    lo,hi=x;assert np.all(lo>=0) and np.all(hi>=lo) and not np.isnan(hi).any()
    finite=np.isfinite(hi);safe=np.where(finite,hi,1.)
    ll,lh=entropy_exact_float(lo);hl,hh=entropy_exact_float(safe)
    lower=np.minimum(ll,hl);upper=np.maximum(lh,hh)
    cross=(lo<=float('0.3678794411714424'))&(hi>=float('0.3678794411714422'))
    upper=np.where(cross,np.maximum(upper,float('0.5307378454230432')),upper)
    return np.where(finite,lower,-np.inf),upper
