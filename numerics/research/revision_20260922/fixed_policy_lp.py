"""SciPy worker: common allocation and syndrome retention, no hidden state input."""
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,vstack,hstack


def main():
    data=np.load(sys.argv[1]);values=data['values']
    alphas=data['alphas'].astype(int);parents=data['parents'].astype(int)
    betas=data['betas'].astype(int);groups=data['groups'].astype(int)
    n=values.shape[1];ng=int(groups.max())+1
    group=coo_matrix((np.ones(len(alphas)),(groups,alphas)),shape=(ng,n)).tocsr()
    relation=coo_matrix((np.r_[np.ones(len(betas)),-np.ones(len(betas))],
                        (np.r_[np.arange(len(betas)),np.arange(len(betas))],np.r_[betas,parents])),shape=(len(betas),n)).tocsr()
    scale=max(float(np.max(np.abs(values))),1e-14)
    constraints=vstack((hstack((group,np.zeros((ng,1)))),
                        hstack((relation,np.zeros((len(betas),1)))),
                        csr_matrix(np.column_stack((-values/scale,np.ones(len(values)))))),format='csr')
    rhs=np.r_[np.ones(ng),np.zeros(len(betas)+len(values))]
    objective=np.r_[np.zeros(n),-1.]
    result=linprog(objective,A_ub=constraints,b_ub=rhs,
                   bounds=[(0,None)]*n+[(None,None)],method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise RuntimeError(result.message)
    np.savez(sys.argv[2],weights=result.x[:-1],dual=-result.ineqlin.marginals[-len(values):],
             upper=result.x[-1]*scale)


if __name__=='__main__':main()
