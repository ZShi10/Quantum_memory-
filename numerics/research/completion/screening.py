"""Safe instrument screening using independently certified incumbents.

The initial per-flag relative entropy is at most concurrence squared.
An instrument whose sum of these bounds is below an achievable incumbent
at its own minimum memory cannot improve any larger memory budget either.
This removes computation, not protocols, from the declared search class.
"""
from common import *
import json
import numpy as np


def incumbents(q,maximum=8):
    path=DATA/'werner_witness_certificates.json'
    out=np.zeros(maximum+1)
    if path.exists():
        for v in json.loads(path.read_text())['results'].values():
            if v.get('source')!='werner_complete_k16.json':continue
            if abs(v['Q']-float(q))<1e-10:
                M=v['M'];lower=np.nextafter(float(v['lower']),-np.inf)
                out[M:]=np.maximum(out[M:],lower)
    return out


def relevant_states(data,aggregation,raw_upper,incumbent,padding=0.):
    bi,si,fi=aggregation
    totals=np.bincount(bi,weights=np.nextafter(fi,np.inf)*raw_upper[si],minlength=len(data['entries']))
    totals=np.nextafter(totals/(1-256*np.finfo(float).eps),np.inf)+padding
    keep=np.array([totals[i]>=incumbent[e['M']] for i,e in enumerate(data['entries'])])
    state_keep=np.zeros(len(raw_upper),dtype=bool)
    state_keep[si[keep[bi]]]=True
    return state_keep,totals,keep
