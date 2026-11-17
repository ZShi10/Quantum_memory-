"""Finite binary algebra and IID code equivalences; no sampling."""
from itertools import combinations, product, permutations
from functools import lru_cache
import numpy as np


def parity(x):
    return int(x).bit_count() & 1


def span(rows):
    out=[0]
    for row in rows: out += [x ^ int(row) for x in out]
    return out


def rref(rows, width):
    rows=list(map(int,rows)); at=0
    for col in range(width):
        pivot=next((j for j in range(at,len(rows)) if rows[j]>>col&1),None)
        if pivot is None: continue
        rows[at],rows[pivot]=rows[pivot],rows[at]
        for j in range(len(rows)):
            if j!=at and rows[j]>>col&1: rows[j]^=rows[at]
        at+=1
    return tuple(rows[:at])


def spaces(n,k):
    for pivots in combinations(range(n),k):
        free=[(r,c) for r,p in enumerate(pivots) for c in range(p+1,n) if c not in pivots]
        for bits in product((0,1),repeat=len(free)):
            rows=[1<<p for p in pivots]
            for (r,c),b in zip(free,bits): rows[r]|=b<<c
            yield tuple(rows)


def symp(a,b,m):
    mask=(1<<m)-1
    return parity((a&mask)&(b>>m)) ^ parity((b&mask)&(a>>m))


def quadratic(a,m):
    return parity((a&((1<<m)-1))&(a>>m))


@lru_cache(None)
def gl_actions(k):
    if k==0:return np.zeros((1,1),dtype=np.int16)
    result=[]
    for rows in permutations(range(1,1<<k),k):
        if len(rref(rows,k))!=k:continue
        result.append([sum(parity(g&x)<<j for j,g in enumerate(rows)) for x in range(1<<k)])
    return np.array(result,dtype=np.int16)


def code_key(n,rows):
    """Quotient by row operations and coordinate permutations, including zeros.

    Use the dual when k>n/2 to keep GL(k,2) enumeration small. A column
    multiset modulo GL is a complete invariant of coordinate equivalence.
    """
    k=len(rows)
    basis=rows
    if k>n//2:
        basis=rref([x for x in range(1<<n) if all(parity(x&r)==0 for r in rows)],n)
    dim=len(basis)
    cols=np.array([sum((r>>j&1)<<i for i,r in enumerate(basis)) for j in range(n)])
    variants=np.sort(gl_actions(dim)[:,cols],axis=1)
    best=min(map(tuple,variants.tolist()))
    return (n,k,best)


def code_library(max_n=6):
    entries=[]; counts={}; orbit_counts={}
    for n in range(1,max_n+1):
        seen={};count=0
        for k in range(1,n+1):
            for rows in spaces(n,k):
                count+=1;key=code_key(n,rows)
                if key not in seen:
                    words=span(rows)
                    seen[key]=dict(n=n,k=k,d=min(x.bit_count() for x in words if x),
                                   rows=list(rows),multiplicity=0)
                seen[key]['multiplicity']+=1
        counts[n]=count;orbit_counts[n]=len(seen)
        for entry in seen.values():
            entry['id']=f"C{len(entries):03d}"
            entry['endpoint']=(entry['k']==1 and entry['d']==n) or (entry['k']==n-1 and entry['d']==2) or n==1
            entry['components']=code_components(n,entry['rows'])
            entry['indecomposable']=len(entry['components'])==1
            entries.append(entry)
    return entries,counts,orbit_counts


def code_components(n,rows):
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    # For RREF rows, connected support components are a direct-sum
    # decomposition. Zero columns give zero-dimensional wasted coordinates.
    for row in rref(rows,n):
        supp=[j for j in range(n) if row>>j&1]
        for j in supp[1:]:parent[find(j)]=find(supp[0])
    return [group for _,group in sorted((a,[j for j in range(n) if find(j)==a]) for a in {find(i) for i in range(n)})]


def compositions(n):
    return [(n-x-y-z,x,y,z) for x in range(n+1) for y in range(n-x+1) for z in range(n-x-y+1)]


def physical_types(m):
    """Pauli vector: low m bits X, high m bits Z."""
    result=[]
    for p in range(1<<(2*m)):
        counts=[0]*4
        for j in range(m):
            x=p>>j&1;z=p>>(m+j)&1
            counts[{(0,0):0,(1,0):1,(1,1):2,(0,1):3}[x,z]]+=1
        result.append(tuple(counts))
    return result
