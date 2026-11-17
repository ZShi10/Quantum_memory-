"""Small isolated SciPy 1.10 worker (system Python 3.8 is sufficient)."""
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack

def main():
    source,target=sys.argv[1:]
    data=np.load(source)
    values=data['values'];groups=data['groups'].astype(int)
    count=len(groups);ng=int(groups.max())+1
    equality=coo_matrix((np.ones(count),(groups,np.arange(count))),shape=(ng,count)).tocsr()
    equality=hstack((equality,np.zeros((ng,1))),format='csr')
    # Scale the objective and constraints for small but meaningful rates.
    scale=max(float(np.max(np.abs(values))),1e-8)
    inequalities=np.column_stack((-values/scale,np.ones(len(values))))
    objective=np.r_[np.zeros(count),-1.]
    result=linprog(objective,A_ub=inequalities,b_ub=np.zeros(len(values)),
                   A_eq=equality,b_eq=np.ones(ng),bounds=[(0,None)]*count+[(None,None)],
                   method='highs',options={'dual_feasibility_tolerance':1e-9,
                                            'primal_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    np.savez(target,weights=result.x[:-1],upper=result.x[-1]*scale,
             dual=-result.ineqlin.marginals,slack=result.ineqlin.residual)

if __name__=='__main__':main()
