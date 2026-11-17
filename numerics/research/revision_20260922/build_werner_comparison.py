"""Build a transparent working figure from actually computed source points."""
from core import *
import csv

fig=ROOT/'figures';fig.mkdir(exist_ok=True)
rec=json.loads((DATA/'werner_v2_full.json').read_text())['results']
sun=json.loads((DATA/'sun_common_full_AD.json').read_text())['results']
original=list(csv.DictReader((DATA/'sun_reconstructed_nine_sizes.csv').open()))
du=json.loads((DATA/'du_Fig15_werner_recomputed.json').read_text())
assert du['complete']
grid=np.linspace(0,1/3,500);rows=[]
for q in grid:
    if q>.3201:continue
    r=min(rec,key=lambda v:abs(v['Q']-q));s=min(sun,key=lambda v:abs(v['Q']-q));o=min(original,key=lambda v:abs(float(v['Q'])-q))
    if abs(r['Q']-q)>1e-13:continue
    assert abs(s['Q']-q)<1e-13 and abs(float(o['Q'])-q)<1e-13
    row=dict(Q=float(q),Sun_original=float(o['nine_size_envelope']),Sun_common=s['results']['M2']['rate'],
             Du_reference=du['results'][f'{q:.12g}']['rate'])
    for M in (1,2,4,8):
        value=r['results'][f'M{M}']['rate']
        if M>=2:value=max(value,row['Sun_common'])
        row[f'M{M}']=value if value>1e-14 else 0.
    rows.append(row)
with (fig/'werner_comparison.csv').open('w',newline='') as file:
    w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
atomic_json(fig/'werner_comparison_provenance.json',dict(points=len(rows),sources=['werner_v2_full.json','sun_common_full_AD.json','sun_reconstructed_nine_sizes.csv','du_Fig15_werner_recomputed.json'],
    meaning='Constructive envelope of registered four-decoder recursion/raw pumps and the strengthened Sun-size control. Does not claim the complete m<=8 fixed-block union optimum at every plotted Q.',
    grid='Actual common-grid calculations, no smoothing/interpolation of rates; displayed Q<=0.3201. Tiny floating positives <=1e-14 suppressed as numerical artifacts, not a positivity claim.'))
tex=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=180mm,paperheight=119mm,margin=8mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\pagestyle{empty}
\begin{document}
\centering
\small\textbf{Werner input: memory and the final classical processing}\par\vspace{2mm}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=13mm},width=77mm,height=72mm,
 xlabel={Input error rate $Q$},ylabel={Secret bits per raw pair},grid=major,
 tick label style={font=\scriptsize},label style={font=\small},
 legend style={font=\scriptsize,draw=none},unbounded coords=jump]
\nextgroupplot[xmin=0,xmax=.32,ymin=0,ymax=1,legend to name=sharedlegend,legend columns=3]
\addplot[gray,densely dotted,thick] table[x=Q,y=Sun_original,col sep=comma]{werner_comparison.csv};\addlegendentry{Sun: original convention}
\addplot[black,dashed,thick] table[x=Q,y=Sun_common,col sep=comma]{werner_comparison.csv};\addlegendentry{Sun sizes + common AD}
\addplot[green!40!black,dashdotted,thick] table[x=Q,y=Du_reference,col sep=comma]{werner_comparison.csv};\addlegendentry{Du Fig.15 menu (recomputed)}
\addplot[cyan!60!black,thick] table[x=Q,y=M1,col sep=comma]{werner_comparison.csv};\addlegendentry{$M\leq1$ constructions}
\addplot[blue!75!black,thick] table[x=Q,y=M2,col sep=comma]{werner_comparison.csv};\addlegendentry{$M\leq2$ constructions}
\addplot[orange!85!black,thick] table[x=Q,y=M4,col sep=comma]{werner_comparison.csv};\addlegendentry{$M\leq4$ constructions}
\addplot[red!75!black,thick] table[x=Q,y=M8,col sep=comma]{werner_comparison.csv};\addlegendentry{$M\leq8$ constructions}
\nextgroupplot[xmin=.18,xmax=.32,ymode=log,ymin=1e-7,ymax=.1,ytick={1e-7,1e-5,1e-3,.1},ylabel={}]
\addplot[gray,densely dotted,thick] table[x=Q,y=Sun_original,col sep=comma]{werner_comparison.csv};
\addplot[black,dashed,thick] table[x=Q,y=Sun_common,col sep=comma]{werner_comparison.csv};
\addplot[green!40!black,dashdotted,thick] table[x=Q,y=Du_reference,col sep=comma]{werner_comparison.csv};
\addplot[cyan!60!black,thick] table[x=Q,y=M1,col sep=comma]{werner_comparison.csv};
\addplot[blue!75!black,thick] table[x=Q,y=M2,col sep=comma]{werner_comparison.csv};
\addplot[orange!85!black,thick] table[x=Q,y=M4,col sep=comma]{werner_comparison.csv};
\addplot[red!75!black,thick] table[x=Q,y=M8,col sep=comma]{werner_comparison.csv};
\end{groupplot}
\end{tikzpicture}\par
\ref{sharedlegend}\par\vspace{2mm}
\scriptsize Ideal IID model; final $X/Y/Z$ permission. Curves join computed points.\par
\scriptsize Constructive envelopes; whole-class optimality is certified separately at selected inputs.
\end{document}
'''
(fig/'werner_comparison.tex').write_text(tex,encoding='utf-8')
print('Working comparison figure data:',len(rows),'actual points',flush=True)
