"""Exact continuous certificates for the selected fixed real BB84 pumps."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
import importlib.util,json,time
from interval_certificate import IV,powpoly


def ratios(eb,ep,r,axis):
    b,p=F(str(eb)),F(str(ep))
    if axis:b,p=p,b
    t=min(b,p);A=(1-b)**r;B=b**r;den=A+B
    v=powpoly([1-b-2*p,1-b-2*p+2*t],r);w=powpoly([b,b-2*t],r)
    cost=(1+sum((1-b)**j+b**j for j in range(1,r)))/den
    out=[([v[j]+w[j] for j in range(r+1)],[den]*(r+1)),
         ([B+w[j] for j in range(r+1)],[den+v[j]+w[j] for j in range(r+1)]),
         ([B-w[j] for j in range(r+1)],[den-v[j]-w[j] for j in range(r+1)]),
         ([1/cost]*(r+1),[F(1)]*(r+1))]
    return [([IV(x) for x in a],[IV(x) for x in b]) for a,b in out]


def main():
    spec=importlib.util.spec_from_file_location('external_repetition_certificate',EXT/'certify_long_repetition.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=json.loads((DATA/'current_results.json').read_text());out={};started=time.time()
    for eb,ep in [(.12,.35),(.24,.24),(.245,.245)]:
        key=f'B_{eb:.5f}_{ep:.5f}_real_M2';record=old['protocols'][key];rule=record['fixed_protocol'];par=rule['parameters']
        rr=ratios(eb,ep,par['r'],par['axis']);target=D(str(record['rate']))*D('.9999')
        pending=[(F(0),F(1),0)];cells=[];minimum=D('Infinity')
        while pending:
            a,b,d=pending.pop();lo=module.bound(rr,a,b,rule['binary_rounds'],rule['final_n'])
            if lo<target:
                assert d<26,(key,a,b,lo,target)
                mid=(a+b)/2;pending.extend([(mid,b,d+1),(a,mid,d+1)]);continue
            minimum=min(minimum,lo);cells.append(dict(interval=[str(a),str(b)],lower=str(lo)))
        out[key]=dict(eb=eb,ep=ep,lower=str(minimum),fixed_protocol=rule,cells=cells)
        print('Fixed pumping certificate',key,str(minimum),'cells',len(cells),'seconds',round(time.time()-started,1),flush=True)
    (DATA/'fixed_pumping_certified.json').write_text(json.dumps(dict(status='passed',arithmetic='Exact Bernstein ratios and 60-digit outward rounding; one fixed ED/AD choice on all compatible z.',results=out),separators=(',',':')))


if __name__=='__main__':main()
