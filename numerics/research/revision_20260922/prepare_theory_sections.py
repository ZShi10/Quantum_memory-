"""Prepare audited theory sections; never overwrite the current main manuscript."""
from core import *


def main():
    old=(PROJECT/'manuscript/main.tex').read_text(encoding='utf-8')
    draft=ROOT/'draft'
    start=old.index(r'\section{Exact memory')
    end=old.index(r'\section{BB84 information and implementable adaptation}')
    theory=old[start:end].replace(r'\section{',r'\subsection{')
    # Credit the existing basis-permutation threshold result explicitly.
    theory=theory.replace('With immediate $Y$ key measurement, full operations',
                          r'As in the basis-permutation observation of Sun et al.\ \cite{Sun}, immediate $Y$ key measurement with full operations')
    theory=theory.replace('The construction does not prove that two qubits are necessary.',
        'The construction makes the register size and the uniform finite parameters explicit; pumping and the recurrence mechanism are established techniques. It does not prove that two qubits are necessary.')
    # Long proofs are placed in appendices to keep the protocol narrative readable.
    proofs=[]
    for label in ('memory','threshold'):
        begin=theory.index(r'\begin{proof}')
        finish=theory.index(r'\end{proof}',begin)+len(r'\end{proof}')
        proofs.append((label,theory[begin:finish]))
        theory=theory[:begin]+r'The proof is given in Appendix~\ref{app:proof-'+label+'}.\n'+theory[finish:]
    (draft/'memory_thresholds.tex').write_text(theory,encoding='utf-8')
    appendix=''
    for label,proof in proofs:
        title='Streaming measurement memory' if label=='memory' else 'Uniform positivity with two live qubits'
        appendix+=r'\section{'+title+r'}\label{app:proof-'+label+'}\n'+proof+'\n\n'
    start=old.index(r'\section{One common finite AD length}')
    finish=old.index(r'\section{Infinite endpoint-code tails')
    appendix+=old[start:finish]
    (draft/'memory_threshold_proofs.tex').write_text(appendix,encoding='utf-8')
    start=old.index(r'\section{Infinite endpoint-code tails')
    finish=old.index(r'\section{Enumeration, pruning')
    (draft/'code_tails_base.tex').write_text(old[start:finish],encoding='utf-8')
    print('Prepared memory, threshold and code-tail sections from the audited old source.')


if __name__=='__main__':main()
