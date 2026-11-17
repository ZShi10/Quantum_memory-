"""Constructive full-hidden-parameter checks of residual Y-terminal choices."""
from pathlib import Path
import json,sys
import full_bb84_control as driver
HERE=Path(__file__).resolve().parent
TASK=HERE/'full_bb84/high_noise_witnesses/0.23'


def main():
    driver.bootstrap(TASK)
    import bb84_joint_extension as engine
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),
                   allocation=1.,allocation_exact='1')]
    rows=[]
    for path in ['RRZ','RRZR']:
        for B in [0,1,2]:
            ed=dict(family='real_decoder_recurrence',path=path,final_axis='Y',depth=len(path),
                    memory_upper_bound=4,minimal_memory_proved=False,successful_ancestry_inputs=2**len(path))
            certificate=engine.certify_policy(.23,.23,ed,B,strategy,atol=2e-9,max_cells=12000)
            row=dict(Q=.23,ED=ed,B=B,strategy=strategy,lower=max(0.,certificate['lower']),certificate=certificate)
            rows.append(row);driver.save(TASK/(path+f'_B{B}.json'),row)
            print(path,B,row['lower'],certificate,flush=True)
    driver.save(TASK/'receipt.json',dict(complete=True,rows=rows,
        best=max(rows,key=lambda r:r['lower']),scope='Six specified Y-terminal homogeneous protocols with Watanabe; full hidden source interval. Feasible witnesses only.'))


if __name__=='__main__':main()
