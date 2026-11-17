"""Independent gate propagation and high-precision graph-family witnesses."""
from pathlib import Path
from fractions import Fraction
from decimal import localcontext
import json,sys,time
import numpy as np
from graph_measurements import graph_circuit
from circuits import apply_gate,measurement_cut
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from protocols import D,dbstep
from linear_codes import exact_rate

def main():
    entries={e['id']:e for e in json.loads((DATA/'graph_family_blocks.json').read_text())['entries']}
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    curves=json.loads((DATA/'werner_joint_k12f.json').read_text())['results']
    selected=[next(v for v in curves[mode] if v['Q']==q) for mode,q in [('full_M3',.28),('full_M4',.30),('full_M2',.32)]]
    results=[];start=time.time()
    for record in selected:
        ed=entries[record['ED']];m=ed['m'];v=ed['logical_dual_vector'];pivot=v.bit_length()-1
        gates=graph_circuit(m,ed['edges'],ed['input_order'],[0]*m)
        cut=measurement_cut(gates,m);assert cut['M']<=ed['M']
        assert cut['M']>=ed['schmidt_memory_lower']
        tensor=np.zeros((1<<(m-1),4,m+1),dtype=int);mask=(1<<m)-1
        for physical in range(1<<(2*m)):
            final=physical
            for g in gates:final=apply_gate(final,g,m)
            x=final&mask;z=final>>m;logical_x=(x>>pivot)&1;logical_z=(z&v).bit_count()%2
            rep=x^(logical_x*v);s=(rep&((1<<pivot)-1))|((rep>>(pivot+1))<<pivot)
            label=3-logical_x if logical_z else logical_x
            weight=((physical&mask)|(physical>>m)).bit_count()
            tensor[s,label,weight]+=1
        assert np.array_equal(tensor,np.array(ed['werner_weights']))
        q=Fraction(str(record['Q']));f=1-Fraction(3,2)*q;t=q/2
        powers=[f**(m-w)*t**w for w in range(m+1)]
        with localcontext() as ctx:
            ctx.prec=130;total=D(0);details=[]
            for branch in record['branches']:
                if branch['contribution']<=0:continue
                raw=[sum(int(c)*w for c,w in zip(row,powers)) for row in tensor[branch['syndrome']]]
                ps=sum(raw);p=tuple(D(x)/D(ps) for x in raw);factor=D(ps)/m
                for _ in range(branch['binary_rounds']):p,s=dbstep(p);factor*=s/2
                code=codes[branch['code']];r=factor*exact_rate(p,code['n'],code['rows'])
                total+=r;details.append({**branch,'rate_130digit':str(r)})
            assert abs(float(total)-record['rate'])<max(1e-18,1e-8*record['rate'])
            result=dict(ED=ed['id'],Q=str(q),rate=str(total),memory_upper=cut['M'],memory_lower=ed['schmidt_memory_lower'],
                        graph=ed['graph'],edges=ed['edges'],input_order=ed['input_order'],logical_dual_vector=v,
                        gates=gates,cut=cut,branches=details,all_Pauli_strings_verified=4**m)
            results.append(result);print(ed['id'],'Q',q,'M bounds',cut['M'],ed['schmidt_memory_lower'],'rate',str(total)[:55],'seconds',time.time()-start,flush=True)
    (DATA/'graph_family_witnesses_130digit.json').write_text(json.dumps(dict(status='passed',results=results),indent=2))

if __name__=='__main__':main()
