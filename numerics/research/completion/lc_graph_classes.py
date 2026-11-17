"""Complete LC + vertex-permutation graph classes, without graph packages.

Induction: put the induced (n-1)-vertex graph in an LC-isomorphism
representative, then add the nth vertex with every possible neighborhood.
Each new component is closed under every local complementation and exact
graph isomorphism. Graph isomorphism uses exhaustive color-constrained
backtracking; fingerprints only prune and never establish equivalence.
"""
from common import *
from collections import deque,Counter
import json,time


def neighbors(mask):
    while mask:
        bit=mask&-mask;yield bit.bit_length()-1;mask^=bit


def complement(graph,v):
    rows=list(graph);nb=rows[v]
    for u in neighbors(nb):rows[u]^=nb^(1<<u)
    return tuple(rows)


def colors(graph):
    n=len(graph)
    initial=[(a.bit_count(),sum((graph[j]&a).bit_count() for j in neighbors(a))//2) for a in graph]
    palette={s:i for i,s in enumerate(sorted(set(initial)))};labels=[palette[s] for s in initial]
    for _ in range(n):
        sig=[(labels[i],tuple(sorted(labels[j] for j in neighbors(a)))) for i,a in enumerate(graph)]
        palette={s:i for i,s in enumerate(sorted(set(sig)))};new=[palette[s] for s in sig]
        if new==labels:break
        labels=new
    fingerprint=tuple(sorted((initial[i],labels[i],tuple(sorted(labels[j] for j in neighbors(a)))) for i,a in enumerate(graph)))
    return labels,fingerprint


def isomorphic(a,b,ca,cb):
    n=len(a);mapping={};used=0
    candidates=[[j for j in range(n) if ca[i]==cb[j]] for i in range(n)]
    def visit():
        nonlocal used
        if len(mapping)==n:return True
        i=min((v for v in range(n) if v not in mapping),key=lambda v:(sum(not(used>>j&1) for j in candidates[v]),-sum((a[v]>>k)&1 for k in mapping)))
        for j in candidates[i]:
            if used>>j&1:continue
            if any(((a[i]>>k)&1)!=((b[j]>>l)&1) for k,l in mapping.items()):continue
            mapping[i]=j;used|=1<<j
            if visit():return True
            used^=1<<j;del mapping[i]
        return False
    return visit()


class Registry:
    def __init__(self):self.graphs=[];self.labels=[];self.components=[];self.buckets={};self.labeled={};self.iso_calls=0
    def add(self,graph,component):
        if graph in self.labeled:return self.labeled[graph],False
        labels,fp=colors(graph)
        for j in self.buckets.get(fp,[]):
            self.iso_calls+=1
            if isomorphic(graph,self.graphs[j],labels,self.labels[j]):
                self.labeled[graph]=j;return j,False
        j=len(self.graphs);self.graphs.append(graph);self.labels.append(labels);self.components.append(component)
        self.buckets.setdefault(fp,[]).append(j);self.labeled[graph]=j;return j,True


def main():
    previous=[()];result={};start=time.time();expected=[1,1,2,4,11,34,156,1044,12346]
    for n in range(1,9):
        registry=Registry();representatives=[]
        for smaller in previous:
            for mask in range(1<<(n-1)):
                graph=tuple([a|(((mask>>j)&1)<<(n-1)) for j,a in enumerate(smaller)]+[mask])
                component=len(representatives);j,new=registry.add(graph,component)
                if not new:continue
                representatives.append(graph);queue=deque([j])
                while queue:
                    at=queue.popleft();g=registry.graphs[at]
                    for v in range(n):
                        nxt,new=registry.add(complement(g,v),component)
                        assert registry.components[nxt]==component
                        if new:queue.append(nxt)
        assert len(registry.graphs)==expected[n],(n,len(registry.graphs),expected[n])
        result[str(n)]=dict(representatives=representatives,unlabeled_graphs=len(registry.graphs),
                           LC_isomorphism_classes=len(representatives),iso_calls=registry.iso_calls,
                           orbit_sizes=dict(Counter(registry.components)))
        previous=representatives
        (DATA/'lc_graph_classes.json').write_text(json.dumps(dict(results=result),separators=(',',':')))
        print('Complete LC classes',n,len(representatives),'unlabeled',len(registry.graphs),'seconds',round(time.time()-start,1),flush=True)
    (DATA/'lc_graph_classes.json').write_text(json.dumps(dict(status='passed',results=result,
        scope='Complete simple-graph LC and isomorphism classes through eight vertices, including disconnected graphs. Exact isomorphism checks and full LC closure; no heuristic equivalence or external classification table used.'),separators=(',',':')))


if __name__=='__main__':main()
