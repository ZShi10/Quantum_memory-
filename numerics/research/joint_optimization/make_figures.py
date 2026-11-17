"""PGFPlots scientific figures; original manuscript is not edited."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)

def number(v):
    if not v:return '--'
    a,b=f'{v:.4e}'.split('e');return rf'${a}\times10^{{{int(b)}}}$'

def main():
    blocks=json.loads((DATA/'werner_joint_k12f.json').read_text())['results']
    pumps=json.loads((DATA/'werner_pumping_joint.json').read_text())['results']
    pumping={round(v['Q'],8):v['combined_rate'] for v in pumps}
    witnesses=json.loads((DATA/'pumping_witnesses_high_precision.json').read_text())['results']
    for v in witnesses:pumping[round(float(v['Q']),8)]=float(v['rate_per_raw_pair'])
    qs=[v['Q'] for v in blocks['full_M1']];cols=[qs]
    for M in range(1,5):
        rs=[v['rate'] for v in blocks[f'full_M{M}']]
        if M>=2:rs=[max(r,pumping.get(round(q,8),0.)) for q,r in zip(qs,rs)]
        cols.append([r if r>0 else np.nan for r in rs])
    np.savetxt(FIG/'werner.dat',np.array(cols).T,header='Q M1 M2 M3 M4',comments='')
    np.savetxt(FIG/'boundary.dat',[[float(v['Q']),float(v['rate_per_raw_pair'])] for v in witnesses],header='Q R',comments='')
    rows=[];bbtable=[]
    for path in sorted(DATA.glob('bb84_pilot_[0-9]*.json')):
        item=json.loads(path.read_text());q=item['eb']
        if q!=item['ep']:continue
        fixed=json.loads((DATA/f'bb84_{q:.5f}_{q:.5f}.json').read_text())['results']['real_M1']['lower']
        values=[fixed]+[item['results'][k]['lower'] for k in ('real_M2','real_M3','full_M1','full_M2')]
        rows.append([q]+[r if r>0 else np.nan for r in values])
        if q<=.23:bbtable.append(f'{q:.2f} & '+' & '.join(number(r) for r in values)+r' \\')
    np.savetxt(FIG/'bb84.dat',rows,header='Q real1 real2 real3 full1 full2',comments='')
    witnessrows='\n'.join(f"{float(v['Q']):.3f} & {v['r']} & {v['binary_AD_rounds']} & {number(float(v['rate_per_raw_pair']))} \\\\" for v in witnesses)
    source=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.5cm]{geometry}
\usepackage{pgfplots,booktabs,amsmath}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18,every axis/.append style={grid=major,grid style={gray!15},tick label style={font=\small},label style={font=\small},legend style={font=\small,draw=none},unbounded coords=jump}}
\definecolor{blueone}{RGB}{25,118,163}
\definecolor{orangeone}{RGB}{219,122,32}
\definecolor{purpleone}{RGB}{119,85,163}
\setlength{\parindent}{0pt}
\begin{document}
{\Large Numerical research record}\hfill 19 September 2026
\par\medskip
\textbf{Werner inputs: memory, rate, and the threshold.}
Rates are secret bits per raw entangled pair, with ideal local operations
and storage, IID inputs, and asymptotic reconciliation. $M$ is the peak
live qubit count \emph{per party}, including the input port. $M=1$ allows
immediate measurement; $M=2$ allows one stored qubit and one input port.
\begin{center}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.5cm},width=8.6cm,height=7.3cm,ymode=log,xlabel={Werner QBER $Q$}]
\nextgroupplot[xmin=.19,xmax=.322,ymin=1e-12,ymax=.05,ylabel={Secret bits / raw pair},title={Constructed rate envelopes},legend pos=south west]
\addplot[gray,mark=o] table[x=Q,y=M1]{werner.dat};\addlegendentry{$M=1$}
\addplot[blueone,mark=square] table[x=Q,y=M2]{werner.dat};\addlegendentry{$M=2$}
\addplot[orangeone,mark=triangle] table[x=Q,y=M3]{werner.dat};\addlegendentry{$M=3$}
\addplot[purpleone,mark=diamond] table[x=Q,y=M4]{werner.dat};\addlegendentry{$M=4$}
\nextgroupplot[xmin=.295,xmax=.334,ymin=1e-72,ymax=1e-4,title={Checked two-qubit witnesses},legend pos=south west]
\addplot[blueone,thick,mark=*] table[x=Q,y=R]{boundary.dat};\addlegendentry{$M=2$}
\addplot[black,densely dashed] coordinates {(.3333333333,1e-72) (.3333333333,1e-4)};\addlegendentry{Separable boundary}
\end{groupplot}\end{tikzpicture}\end{center}
The left envelope combines all compiled four-input blocks, selected graph
blocks with five to eight inputs, and restart pumping. AD uses the stated
binary prefixes and codes of length at most six. These are achievable
families, not a converse for all fixed-memory protocols. The right panel
uses independent high-precision checks; a positive rate near $1/3$ can
still be extremely small.
\begin{center}\begin{tabular}{rrrr}\toprule
$Q$ & Pumping inputs $r$ & Binary AD rounds & Checked rate \\\midrule
@@WITNESSES@@
\bottomrule\end{tabular}\end{center}
All failed pumping attempts are charged. The mean retry cost differs from
the successful-path input count $r$. The construction proves a positive
rate at every $Q<1/3$ with $M=2$, and does not assert positive rate at $1/3$.
\par\bigskip
\textbf{A useful ablation.} For four-input blocks at $Q=0.28$ and $M\leq2$,
maximizing output fidelity gives $F=0.664384$ and rate
$3.19289\times10^{-5}$. Optimizing the final rate instead gives the lower
fidelity $F=0.651707$ but rate $1.68163\times10^{-4}$, about 5.27 times
higher, in the same declared ED/AD class.
\newpage
{\Large BB84 compatible inputs: no Werner twirl}
\par\medskip
Every compatible state $(1-e_b-e_p+z,e_b-z,z,e_p-z)$ is included, with
$0\leq z\leq\min(e_b,e_p)$. Bounds minimize continuously over this
interval. Allowed pilot tests may identify $z$ before selecting ED and AD;
the real $M=1$ baseline uses a valid fixed strategy because $X/Z$ product
measurements do not identify $z$.
\begin{center}\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.5cm},width=8.6cm,height=7.3cm,ymode=log,xmin=.095,xmax=.247,ymin=1e-8,ymax=.3,xlabel={$e_b=e_p$},legend pos=south west]
\nextgroupplot[title={Real $H$/CNOT operations},ylabel={Worst-case bits / raw pair}]
\addplot[gray,mark=o] table[x=Q,y=real1]{bb84.dat};\addlegendentry{$M=1$}
\addplot[blueone,mark=square] table[x=Q,y=real2]{bb84.dat};\addlegendentry{$M=2$}
\addplot[orangeone,mark=triangle] table[x=Q,y=real3]{bb84.dat};\addlegendentry{$M=3$}
\nextgroupplot[title={Full Clifford, with free $Y$}]
\addplot[gray,mark=o] table[x=Q,y=full1]{bb84.dat};\addlegendentry{$M=1$}
\addplot[blueone,mark=square] table[x=Q,y=full2]{bb84.dat};\addlegendentry{$M=2$}
\end{groupplot}\end{tikzpicture}\end{center}
\textbf{Finite scope:} nonadaptive production Clifford blocks with $m\leq4$,
single-round linear codes with $n\leq6$, and a fixed classical decision
rule after allowed pilot tests. Real-gate memory is a certified
implementation bound. All full-Clifford terminal measurements with
$m\leq4$ have an implementation with $M\leq2$.
\begin{center}\small\begin{tabular}{rrrrrr}\toprule
$Q$ & Real $M=1$ & Real $M=2$ & Real $M=3$ & Full $M=1$ & Full $M=2$ \\\midrule
@@BBTABLE@@
\bottomrule\end{tabular}\end{center}
Missing high-noise points mean this finite calculation supplies no plotted
positive lower bound. Longer AD and pumping are outside this plot. With
free $Y$ measurement, the $25\%$ BB84 supremum is already reachable without
coherent storage. Memory advantages must be stated in terms of rates,
bounded processing resources, or a separately restricted operation set.
\par\medskip
\textbf{Two different code questions.} For a fixed Bell input and one AD
round, interval witnesses and analytic infinite-tail bounds prove that
$[6,4,2]$ and $[6,3,3]$ codes can beat every repetition/SPC length.
Adding and reoptimizing preceding B rounds changes that input: SPC lengths
52 and 64 then reverse the two selected full-protocol comparisons.
Thus single-round endpoint optimality is false in general, while endpoint
optimality for arbitrary multistage processing remains open.
\end{document}
'''
    source=source.replace('@@WITNESSES@@',witnessrows).replace('@@BBTABLE@@','\n'.join(bbtable))
    (FIG/'research_numerics.tex').write_text(source,encoding='utf-8')
    print('Saved PGFPlots source and numerical tables.')

if __name__=='__main__':main()
