"""Integrate the new operation comparison without touching the prior PDF/ZIP."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'revised'
def replace_span(path,start,end,replacement):
    text=path.read_text(encoding='utf-8')
    if replacement.strip() in text:return
    left=text.index(start);right=text.index(end,left)
    backup=HERE/'manuscript_before_integration'/path.relative_to(ROOT)
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_text(text,encoding='utf-8')
    path.write_text(text[:left]+replacement+'\n'+text[right:],encoding='utf-8')

replace_span(ROOT/'main_text.tex',
    r'\subsection{Does the advantage survive additional BB84 operations?}',
    r'\subsection{Target memory with $X/Z$ key readout}',
    '\\input{sections/main_clifford}\n')
replace_span(ROOT/'sections/s5_enumeration.tex',
    r'\subsection{Full-Clifford BB84 comparison}',
    r'\subsection{Comparison with Jansen and preservation of execution data}',
    '\\input{sections/s5_full_clifford}\n')

path=ROOT/'main.tex';text=path.read_text(encoding='utf-8')
old='''For BB84-compatible information at $Q=0.17$, a three-qubit
construction also exceeds the two-qubit upper of an enlarged finite
family that includes complete five-input Clifford readouts and all
terminal Pauli axes, by more than $5.81\\%$.
For an explicitly enlarged BB84 family, constructive continuum lower
bounds and finite-family upper certificates determine the memory needed
to reach selected nonzero rates.'''
new='''For BB84-compatible information, we also include every two-input
Clifford decoder within specified recursive constructions. At $Q=0.17$,
the additional operations reduce the memory required for a target rate;
at $Q=0.20$, three qubits still outperform the complete two-qubit
family by more than $30.02\\%$. This comparison includes full-Clifford
terminal measurements through five inputs, the complete stated
classical code menu, and every compatible hidden correlation.
Constructive lower bounds and finite-family upper bounds determine
the memory needed to reach selected nonzero rates.'''
assert old in text or new in text
text=text.replace(old,new);path.write_text(text,encoding='utf-8')

path=ROOT/'sections/s1_model.tex';text=path.read_text(encoding='utf-8')
old='''$\\mathcal P^{(+)}$ also admits all terminal Pauli axes and complete
full-Clifford blocks through five inputs, as specified in
Sec.~\\ref{sec:full-bb84}. These permissions are kept fixed within every
comparison; the BB84 compatible-source interval is unchanged.'''
new='''$\\mathcal P^{\\rm Cl}$ admits all terminal Pauli axes, complete
full-Clifford blocks through five inputs, and every two-input Clifford
map at recursive nodes within the stated structural cutoffs.
Its matched real-node control restricts the recursive maps while
keeping the same full small-block component. Sections~\\ref{sec:full-bb84}
and \\ref{sec:clifford-nodes} specify these permissions; the compatible
BB84 source interval is unchanged.'''
assert old in text or new in text
path.write_text(text.replace(old,new),encoding='utf-8')
path=ROOT/'sections/s9_numerical_appendix.tex';text=path.read_text(encoding='utf-8')
text=text.replace('enlarged-axis finite\ncomparison $\\mathcal P^{(+)}$',
                  'full-Clifford finite\ncomparison $\\mathcal P^{\\rm Cl}$')
path.write_text(text,encoding='utf-8')
print('Integrated full-Clifford comparison; numerical tables and figures follow separately.')
