"""Independent outward upper/lower bounds for the selected controlled examples."""
from core import *
from fractions import Fraction as F
from compound_menu_bounds import menu_bounds
from compound_terminals import actions
from certify_bb84_recursive import step
from certify_recursive import prepare
from werner_ablations import exact_sun_output
from sun_common_objective import outputs
import stable_intervals as iv


def series(state,y,K=16):
    pp=[];yy=[]
    for _ in range(K+1):
        pp.append(state);yy.append(y);state,s=step(state,state,'Z');y=iv.scale(iv.mul(y,s),F(1,2))
    return tuple(np.concatenate([v[j] for v in pp]) for j in (0,1)),tuple(np.concatenate([v[j] for v in yy]) for j in (0,1))


def exact_boxes(values):
    vals=[iv.rational(v) for v in values]
    return tuple(np.array([[v[j] for v in vals]]) for j in (0,1))


def main():
    menu=actions(128);path=DATA/'controlled_examples_certified.json';out={}
    src=json.loads((DATA/'werner_fixed_ED_ablations.json').read_text())['results']['0.3_M4']
    p,y,_=prepare(F(3,10),src['ED'])
    pp=(np.array([[max(0.,float(iv.down(float(v.lo)))) for v in p]]),np.array([[float(iv.up(float(v.hi))) for v in p]]))
    yy=(np.array([max(0.,float(iv.down(float(y.lo))))]),np.array([float(iv.up(float(y.hi)))]))
    pp,yy=series(pp,yy);lo,hi=menu_bounds(pp)
    lo=iv.down(lo*yy[0][:,None]);hi=iv.up(hi*yy[1][:,None])
    no=[i for i,a in enumerate(menu) if a['mode']=='no_otp'];full=[i for i,a in enumerate(menu) if a['kind']!='Watanabe_2007']
    out['compressed_OTP_fixed_ED']=dict(Q=.3,M=4,ED=src['ED'],
           no_OTP=dict(lower=float(lo[:,no].max()),upper=float(hi[:,no].max())),
           all_Du_modes=dict(lower=float(lo[:,full].max()),upper=float(hi[:,full].max())),
           scope='Same ED and final basis, every B<=16 and registered Du code. All terms evaluated outward; no reoptimization of ED.')
    atomic_json(path,out);print('OTP control',out['compressed_OTP_fixed_ED'],flush=True)
    control=json.loads((DATA/'fidelity_control_current_menu.json').read_text());_,_,_,records=outputs(.28)
    tests=[]
    for row in records:
        if row.get('m')!=3:continue
        vals,y=exact_sun_output(.28,row)
        if max(vals)!=F(control['highest_fidelity_exact']):continue
        yy0=iv.rational(y);tests.append(series(exact_boxes(vals),tuple(np.array([v]) for v in yy0)))
    pp=tuple(np.concatenate([v[0][j] for v in tests]) for j in (0,1));yy=tuple(np.concatenate([v[1][j] for v in tests]) for j in (0,1))
    lo,hi=menu_bounds(pp);upper=float(np.maximum(0,iv.up(hi*yy[1][:,None])).max())
    out['fidelity_first']=dict(**control,fidelity_first_entire_menu_upper=upper,
          certified_separation=control['joint']['lower']>upper)
    atomic_json(path,out);print('Fidelity whole-control upper',upper,'joint lower',control['joint']['lower'],flush=True)


if __name__=='__main__':main()
