"""Independent arbitrary-precision retry and near-boundary rate witnesses."""
from pathlib import Path
from decimal import localcontext
import sys,json
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
sys.path.insert(0,str(ROOT.parents[1]/'manuscript'/'code'))
from protocols import D,dbstep
from linear_codes import exact_rate

def main():
    records=json.loads((DATA/'werner_pumping_joint.json').read_text())['results']
    codes={e['id']:e for e in json.loads((DATA/'codes.json').read_text())['entries']}
    output=[]
    for record in records:
        if record['Q'] not in (.3,.32,.325,.33,.332):continue
        with localcontext() as ctx:
            ctx.prec=180;q=D(record['Q']);r=record['r']
            A=(1-q)**r;B=q**r;V=(1-2*q)**r;total=A+B
            p=((A+V)/(2*total),(A-V)/(2*total),B/(2*total),B/(2*total))
            cost=(1+sum((1-q)**j+q**j for j in range(1,r)))/total
            factor=1/cost
            for _ in range(record['binary_rounds']):p,s=dbstep(p);factor*=s/2
            code=codes[record['code']];secret=exact_rate(p,code['n'],code['rows']);rate=factor*secret
            assert rate>0
            relative_error=abs(float(rate)/record['rate']-1)
            # The conditional entropy gain can be very small near 1/3.
            # Check an absolute entropy-scale residual and retain the
            # high-precision rate instead of promising relative binary64
            # accuracy after cancellation.
            assert abs(float(secret)-record['rate']/float(factor))<1e-12
            witness=dict(Q=str(q),r=r,M=2,binary_AD_rounds=record['binary_rounds'],code=code,
                         expected_raw_cost=str(cost),AD_input=[str(x) for x in p],conditional_code_rate=str(secret),rate_per_raw_pair=str(rate),
                         precision=180,entropy_precision=130,search_rate_relative_error=relative_error)
            output.append(witness);print('Q',q,'r',r,'rate',str(rate)[:65],flush=True)
    (DATA/'pumping_witnesses_high_precision.json').write_text(json.dumps(dict(status='passed',results=output),indent=2))

if __name__=='__main__':main()
