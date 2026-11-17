"""Generate publication figures, certified resource tables and review report."""
from common import *
import json,csv,math
from pathlib import Path

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def number(x,digits=3):
    if x<=0:return '--'
    a,b=f'{x:.{digits}e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
def table(headers,rows):
    return r'\begin{center}\begingroup\small\setlength{\tabcolsep}{4pt}\begin{tabular}{'+'l'*len(headers)+r'}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(row)+r'\\' for row in rows)+r'\bottomrule\end{tabular}\endgroup\end{center}'
def plot(curves,xlabel,xrange,yrange,width='7.4cm',height='6.1cm',title=''):
    s=rf'\begin{{tikzpicture}}\begin{{axis}}[width={width},height={height},ymode=log,xmin={xrange[0]},xmax={xrange[1]},ymin={yrange[0]},ymax={yrange[1]},xlabel={{{xlabel}}},ylabel={{Bits / raw pair}},title={{{title}}},grid=major,grid style={{gray!15}},legend style={{font=\scriptsize,draw=none}},legend pos=south west,tick label style={{font=\scriptsize}},label style={{font=\small}},title style={{font=\small}},unbounded coords=jump]'
    for label,points,style in curves:
        coords=' '.join(f'({x:.6g},{y:.12g})' for x,y in points if y>0)
        s+=rf'\addplot[{style},thick,mark size=1.6pt] coordinates {{{coords}}};\addlegendentry{{{label}}}'
    return s+r'\end{axis}\end{tikzpicture}'
def h(x):return -x*math.log2(x)-(1-x)*math.log2(1-x) if 0<x<1 else 0.
def hinv(y):
    a,b=0.,.5
    for _ in range(70):
        m=(a+b)/2
        if h(m)<y:a=m
        else:b=m
    return (a+b)/2


def main():
    d=read(DATA/'current_results.json');w={(r['Q'],r['budget']):r for r in d['six_state']};bb={(r['eb'],r['ep'],r['permissions'],r['budget']):r for r in d['bb84']};cat=d['protocols']
    qs=sorted({x[0] for x in w});bqs=sorted({x[0] for x in bb if x[0]==x[1]})
    short={r['Q']:r['results'] for r in read(DATA/'werner_complete_k16.json')['results']};long={r['Q']:r['results'] for f in DATA.glob('werner_complete_long*.json') for r in read(f)['results']}
    styles={1:'gray,mark=o',2:'blue!70!black,mark=square',3:'orange!85!black,mark=triangle',4:'green!50!black,mark=diamond',5:'violet,mark=+',6:'red!70!black,mark=star',8:'black,dashed,mark=x'}
    coreplots=[]
    for src,title in [(short,'Complete quantum class; $n\\leq7$'),(long,'Same quantum class; endpoint $n\\leq128$')]:
        curves=[(f'$M\\leq{M}$',[(q,row[f'M{M}']['rate']) for q,row in sorted(src.items())],styles[M]) for M in (1,2,3,4)]
        coreplots.append(plot(curves,'Werner $Q$',(.195,.321),(3e-15,.04),title=title))
    corefig=r'\begin{center}'+''.join(coreplots)+r'\end{center}'
    comparison=read(DATA/'controlled_comparisons.json')['matched_size_baseline']
    curves=[(f'$M\\leq{M}$',[(q,w[q,M]['rate']) for q in qs],styles[M]) for M in (2,3,4,5,6,8)]
    curves.append(('Matched-size ED + repetition',[(r['Q'],r['rate']) for r in comparison],'gray,densely dotted,mark=o'))
    expanded=plot(curves,'Werner $Q$',(.195,.321),(5e-13,.04),width='14.5cm',height='6.5cm',title='Achievable envelope, including restart pumping and larger graphs')
    bbplots=[]
    baseline={r['Q']:float(r['lower']) for r in read(DATA/'real_M1_baseline_certified.json')['results']}
    for permission in ('real','full'):
        memories=(2,3,8) if permission=='real' else (1,2,3,8)
        curves=[(f'$M\\leq{M}$',[(q,bb[q,q,permission,M]['lower']) for q in bqs],styles[M]) for M in memories]
        if permission=='real':curves.insert(0,('Fixed repetition, $M=1$',sorted(baseline.items()),styles[1]))
        bbplots.append(plot(curves,'$e_b=e_p$',(.095,.247),(1e-28 if permission=='real' else 1e-12,.25),title=permission.capitalize()+' permissions'))
    bbfig=r'\begin{center}'+''.join(bbplots)+r'\end{center}'
    paper=r'''\begin{figure}[tbp]
@@CORE@@
\caption{Werner point optima in the declared finite libraries. Left: all terminal Clifford statistics through eight inputs, all linear codes through length seven, and at most sixteen pooled B rounds. Right: every repetition and SPC length through 128 is added without an ED shortlist. Short-core optimum enclosures and selected-witness certificates are stored separately. Lines connect computed points and do not assert optimality between them.}\label{fig:core}
\end{figure}
\begin{figure}[tbp]\centering
@@EXPANDED@@
\caption{Achievable Werner rates after also including restart pumping and frozen larger graph constructions. The quantum and classical limits now differ from the finite core; this is not a global fixed-memory optimum. Budgets five through eight may coincide. Every positive selected point has an outward-rounded witness certificate.}\label{fig:werner}
\end{figure}
\begin{figure}[tbp]
@@BB@@
\caption{BB84-compatible input, with the entire unknown correlation interval certified at each physical error point. Full permissions include free $Y$ measurement; real permissions use $H$, CNOT, Pauli and $X/Z$ measurements. The real $M=1$ curve is a fixed repetition baseline, not an oracle selected from hidden $z$. Other curves include legal pilot rules and fixed pumping constructions. Lines guide the eye; the physical intervals certified by noise homogenization are listed separately.}\label{fig:bb}
\end{figure}
'''.replace('@@CORE@@',corefig).replace('@@EXPANDED@@',expanded).replace('@@BB@@',bbfig)
    (FIG/'paper_results.tex').write_text(paper,encoding='utf-8')
    # Every interval is operationally realized by the right target, not interpolation.
    regions=[]
    for M in range(1,9):
        prev=0.
        for q in qs:
            r=w[q,M];rec=cat[r['protocol']]
            if r['rate']>0 and rec.get('certified_lower'):
                regions.append(dict(model='Werner',permissions='real_or_full',budget=M,lower_error=prev,upper_error=q,phase_upper=q,rate_lower=r['rate'],protocol=r['protocol'],mapping='Local depolarization to right endpoint'))
            prev=q
    for (b,p,permission,M),r in sorted(bb.items()):
        if r['lower']>0:regions.append(dict(model='BB84_rectangle',permissions=permission,budget=M,lower_error=0,upper_error=b,phase_upper=p,rate_lower=r['lower'],protocol=r['protocol'],mapping='Independent local X/Z noise to target marginals'))
    with (DATA/'physical_error_regions.csv').open('w',newline='',encoding='utf-8') as f:
        ww=csv.DictWriter(f,fieldnames=list(regions[0]));ww.writeheader();ww.writerows(regions)
    thresholds=[];table_rows=[]
    selections=[('Werner','real_or_full',2),('Werner','real_or_full',5),('BB84','real',2),('BB84','full',1),('BB84','full',3)]
    for rate in (1e-2,1e-3,1e-4,1e-5,1e-6,1e-8):
        line=[number(rate,0)]
        for model,permission,M in selections:
            feasible=[(q,w[q,M]['rate'],w[q,M]['protocol']) for q in qs if w[q,M]['rate']>=rate and cat[w[q,M]['protocol']].get('certified_lower')] if model=='Werner' else [(q,bb[q,q,permission,M]['lower'],bb[q,q,permission,M]['protocol']) for q in bqs if bb[q,q,permission,M]['lower']>=rate]
            best=max(feasible,default=(0.,0.,''));upper=(2/3 if model=='Werner' else .5)*hinv(1-rate)
            thresholds.append(dict(model=model,permissions=permission,budget=M,target_rate=rate,achievable_Q_lower=best[0],universal_Q_upper=upper,protocol=best[2],actual_rate_lower=best[1]))
            line.append(f'{100*best[0]:.1f}\\%')
        table_rows.append(line)
    with (DATA/'useful_rate_thresholds.csv').open('w',newline='',encoding='utf-8') as f:
        ww=csv.DictWriter(f,fieldnames=list(thresholds[0]));ww.writeheader();ww.writerows(thresholds)
    useful=table(['$r_0$','W, $M=2$','W, $M=5$','BB real, $M=2$','BB full, $M=1$','BB full, $M=3$'],table_rows)
    (FIG/'useful_rates.tex').write_text(useful,encoding='utf-8')
    wrows=[]
    for q in (.20,.23,.28,.29,.30,.305,.31,.315,.32):
        a,b=w[q,2],w[q,8];r=cat[b['protocol']]
        label=b['ED'].replace('LARGE_16_random_2','L16R2').replace('LARGE_20_random_1','L20R1')
        wrows.append([f'{q:.3f}',number(a['rate']),number(b['rate']),str(b['memory']),str(b['inputs']),r'\texttt{'+label.replace('_',r'\_')+'}'])
    brows=[]
    for q in (.18,.20,.21,.22,.23,.24,.245):
        brows.append([f'{q:.3f}']+[number(bb[q,q,p,M]['lower']) for p,M in [('real',2),('real',8),('full',1),('full',8)]])
    asym=[]
    for b,p in ((.10,.30),(.18,.25),(.12,.35)):
        asym.append([f'({b:.2f},{p:.2f})']+[number(bb[b,p,permission,M]['lower']) for permission,M in [('real',2),('real',8),('full',2),('full',8)]])
    report=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,amsmath,amssymb,booktabs,pgfplots}
\usepackage[colorlinks=true,urlcolor=blue!60!black]{hyperref}
\pgfplotsset{compat=1.18}\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\begin{document}
{\LARGE Bounded-memory QKD: completed research comparisons}\hfill 20 September 2026

\textbf{Question.} With stated input information, gate permissions, quantum register size and classical processing limits, which ED+AD construction gives the best final key yield?

All rates below are bits per original pair. Inputs are IID Bell-diagonal, gates and storage are ideal, and extraction is asymptotic. The live register $M$ counts the incoming qubit and all working qubits at each party. It differs from the total ED input count $m$.

\textbf{Three levels of conclusions}
\begin{enumerate}
\item \textbf{Analytic feasibility:} real $M=2$ attains positive rate for every Werner $Q<1/3$ and uniformly for every BB84-compatible family with $e_b+e_p<1/2$. One common finite pumping and AD choice suffices. Endpoints contain separable states. These yields can be extremely small.
\item \textbf{Complete finite classes:} all fixed Clifford terminal instruments through eight Werner inputs or four BB84-compatible inputs; every linear-code class through length seven. Werner permits up to sixteen pooled B rounds; the BB84 core permits a single AD round and a legal pilot. Stored lower and upper bounds enclose these optima.
\item \textbf{Additional achievable rates:} every repetition/SPC length through 128 is searched on the complete Werner quantum library; restart pumping and frozen larger graphs are compared separately. Their envelope is a construction bound, not a capacity theorem.
\end{enumerate}

\textbf{Resource correction.} A complete stabilizer measurement has minimum memory
\[
M_{\min}=1+\min_{\mathrm{orders}}\max_A E(A).
\]
The proof is a Schmidt-rank lower bound and a matching signed streaming compiler. Real measurements need only $H$/CNOT. All 2448 bases through four inputs fit $M\leq2$; the complete eight-input Werner library contains 6895 rate instruments and fits $M\leq4$. New frozen graphs through twenty inputs have actual costs reaching $M=8$.

\textbf{Unknown BB84 correlation.} Every rate certificate covers $z\in[0,\min(e_b,e_p)]$ without a Werner twirl. A real $M=2$ Bell-measurement pilot or full $M=1$ $Y$ pilot permits parameter-dependent production rules. Real $M=1$ cannot use that oracle. A finite-confidence pilot argument covers rule boundaries.

\textbf{Meaning of optimality.} Quantum recursion, heterogeneous pooling, larger arbitrary Clifford circuits and general multistage AD are outside the finite cores. An optimum at a sampled physical error is not a continuous optimality theorem. The protocol-region tables instead give explicit guaranteed intervals/rectangles using local noise homogenization.

The companion manuscript contains the proofs and bibliographic positioning. Machine-readable definitions, all decision cells, code generators, streaming circuits and numerical certificates are in \texttt{research/completion/data}.
\newpage
{\Large Werner input: fixed-class optima and larger constructions}
@@CORE@@
The left panel has rigorous finite-core optimum enclosures. The right panel adds all endpoint lengths through 128; its point search has stated floating pruning tolerances and its selected protocols have independent outward certificates. Lines guide the eye.
@@WTABLE@@
The table includes pumping and larger graph constructions. ``Used $M$'' gives the actual register of the selected protocol at budget eight. Increasing the permitted budget can leave the selected register unchanged.
\newpage
{\Large Werner input: the final achievable envelope}
\begin{center}@@EXPANDED@@\end{center}
Every positive table entry has a directed-rounding certificate. At high noise, classical processing and the restart cost can change which circuit wins. The complete finite-core comparison must therefore be kept separate from this expanded envelope.

\textbf{Controlled fidelity comparison.} At $Q=0.28$, three ED inputs and budget two, the best final rate among maximum-fidelity EDs is $7.0653\times10^{-6}$. Direct final-rate selection yields $1.0829\times10^{-4}$. Both use zero ED syndrome, all codes $n\leq7$, and the same sixteen-round B-prefix cap. This is a controlled example of why the output state, success probability and final code must be optimized jointly.

\textbf{Code separation in a stated class.} At a fixed key basis, an explicit $[6,4,2]$ code gives rate greater than $0.01439497134150498$ on a reachable Bell input; every single-round repetition/SPC code of every length is below $0.01383047967221880$. Exact finite sums and analytic infinite-length tails prove this separation. Reoptimizing the basis or ED changes the question; the unrestricted joint conjecture and general CRO threshold are not settled by this witness.

\textbf{Matched-size baseline.} The data also recompute zero-syndrome Clifford ED plus repetition AD for the nine $(m,n,1)$ sizes considered by Sun et al., using the same joint entropy and raw-pair normalization. This is a precisely defined matched-size implementation, not a claim to reproduce every convention of the original plotting code.
\newpage
{\Large BB84-compatible inputs: permissions and memory}
@@BB@@
@@BTABLE@@
The fixed real $M=1$ repetition baseline needs no hidden-parameter identification. Allowing $Y$ measurement changes the memory-free comparison and already permits the $25\%$ positivity supremum. It must not be presented as a threshold gain caused by memory.
@@ASYM@@
The non-symmetric rows also cover the whole compatible input family. The exceptionally small real rate at $(0.12,0.35)$ is a fixed pumping construction with a whole-interval certificate.
\newpage
{\Large Nonzero-rate requirements and complete protocol records}

The following are \emph{achievable lower endpoints} for the largest tolerated symmetric error at a requested nonzero rate. They come from certified targets on the registered grid, not from treating the final positive grid point as a threshold.
@@USEFUL@@
For each target, local noise maps every smaller Werner error or BB84 marginal pair into the certified target input/family. The same circuit and decision rule therefore give an operational continuous interval/rectangle. This may sacrifice rate in the interior; no optimality between grid points is asserted.

A separate universal upper endpoint is $(2/3)h_2^{-1}(1-r_0)$ for Werner and $(1/2)h_2^{-1}(1-r_0)$ for symmetric BB84. These follow from the relative-entropy key bound and are generally loose. They are included in the CSV without being called fixed-memory converses.

\textbf{Reconstruction.} \texttt{protocol\_catalog.json} contains the selected ED stabilizers or graph/dual vector, current streaming schedule, signed output reconstruction, code generator rows and AD acceptance conventions. BB84 records link every certified decision cell in $t=z/\min(e_b,e_p)$. The physical-error range table is a separate file.

\textbf{Validation.} Signed integer replay checks all compiled graph schedules; 2808 independent dense checks validate the small measurement compiler. Exact rational probability polynomials and outward-rounded entropy arithmetic certify selected protocols. A separate binary64 interval engine encloses the finite cores using analytic logarithm remainders. Long-SPC recursion is cross-checked against independent type enumerators, including length 128.

\textbf{Boundaries of this result.} Ideal IID security and per-pair yield are the assumptions. A general CRO converse, arbitrary infinite-protocol fixed-register optimality, gate/storage noise, clock rates and finite-key/coherent-attack analysis remain outside these claims.
\end{document}
'''
    substitutions={'@@CORE@@':corefig,'@@EXPANDED@@':expanded,'@@BB@@':bbfig,'@@WTABLE@@':table(['$Q$',r'$M\leq2$',r'$M\leq8$','Used $M$','$m$','Selected ED'],wrows),'@@BTABLE@@':table(['$e_b=e_p$',r'Real $M\leq2$',r'Real $M\leq8$',r'Full $M\leq1$',r'Full $M\leq8$'],brows),'@@ASYM@@':table(['$(e_b,e_p)$',r'Real $M\leq2$',r'Real $M\leq8$',r'Full $M\leq2$',r'Full $M\leq8$'],asym),'@@USEFUL@@':useful}
    for a,b in substitutions.items():report=report.replace(a,b)
    (FIG/'research_report.tex').write_text(report,encoding='utf-8')
    print('Generated figures,',len(regions),'certified physical regions and',len(thresholds),'nonzero-rate threshold bounds',flush=True)


if __name__=='__main__':main()
