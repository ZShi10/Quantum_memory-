from common import *
import json,time
import numpy as np
from rates import CodeKernel


def load():
    cache=DATA/'kernels';cache.mkdir(exist_ok=True)
    entries=[e for e in json.loads((DATA/'codes.json').read_text())['entries'] if e['indecomposable']]
    kernels=[]
    for e in entries:
        path=cache/(e['id']+'.npz')
        if path.exists():
            arrays=np.load(path);k=CodeKernel.__new__(CodeKernel)
            k.entry=e;k.n=e['n'];k.k=e['k']
            for key in ('types','polys','hist','multiplicity','marginal'):setattr(k,key,arrays[key])
            k.syndrome_classes=json.loads(str(arrays['syndrome_classes']))
        else:
            k=CodeKernel(e)
            np.savez_compressed(path,types=k.types,polys=k.polys,hist=k.hist,multiplicity=k.multiplicity,
                                marginal=k.marginal,syndrome_classes=json.dumps(k.syndrome_classes))
        kernels.append(k)
    return kernels


if __name__=='__main__':
    start=time.time();ks=load();print('Cached',len(ks),'indecomposable/selected kernels in',time.time()-start,'seconds')
