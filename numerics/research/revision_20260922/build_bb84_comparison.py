"""BB84 achievable curves, with separate certified finite-class anchors."""
from core import *
import csv


def main():
    fig=ROOT/'figures';src=json.loads((DATA/'bb84_policy_curves_real.json').read_text())
    du=json.loads((DATA/'du_Fig15_bb84_recomputed.json').read_text());assert du['complete']
    rows=[]
    for item in sorted(src['results'].values(),key=lambda x:x['b']):
        if item['b']!=item['p']:continue
        q=item['b'];h=0. if q==0 else -q*np.log2(q)-(1-q)*np.log2(1-q)
        control=du['results'][f'{q:.12g}'];assert control['Q']==q
        row=dict(Q=q,one_way=max(0.,1-2*h),Du_reference=control['lower'])
        for M in (1,2,4,8):row[f'M{M}']=max(item['results'][f'M{M}']['lower'],control['lower'])
        rows.append(row)
    with (fig/'bb84_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    anchors=[]
    for q in (.15,.18,.2,.22,.24):
        v=json.loads((DATA/f'bb84_joint_extension_real_{q:.5f}_{q:.5f}_n128_menu_certified.json').read_text())
        anchors.append(dict(Q=q,**{f'M{M}':v['results'][f'M{M}']['lower'] for M in (1,2,4,8)}))
    with (fig/'bb84_anchors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(anchors[0]));writer.writeheader();writer.writerows(anchors)
    tex=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=180mm,paperheight=119mm,margin=8mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\pagestyle{empty}
\begin{document}
\centering\small\textbf{BB84 information: one fixed policy for all compatible correlations}\par\vspace{2mm}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=13mm},width=77mm,height=72mm,
 xlabel={Symmetric error $Q=b=p$},ylabel={Guaranteed bits per raw pair},grid=major,
 tick label style={font=\scriptsize},label style={font=\small},
 legend style={font=\scriptsize,draw=none},unbounded coords=jump]
\nextgroupplot[xmin=0,xmax=.25,ymin=0,ymax=1,legend to name=sharedlegend,legend columns=3]
'''
    colors={1:'cyan!60!black',2:'blue!75!black',4:'orange!85!black',8:'red!75!black'}
    for panel in (0,1):
        if panel:tex+=r'\nextgroupplot[xmin=.10,xmax=.25,ymode=log,ymin=1e-7,ymax=.2,ytick={1e-7,1e-5,1e-3,.1},ylabel={}]'+'\n'
        tex+=r'\addplot[gray,densely dotted,thick] table[x=Q,y=one_way,col sep=comma]{bb84_comparison.csv};'+'\n'
        if not panel:tex+=r'\addlegendentry{One-way $[1-2h_2(Q)]_+$}'+'\n'
        tex+=r'\addplot[green!40!black,dashdotted,thick] table[x=Q,y=Du_reference,col sep=comma]{bb84_comparison.csv};'+'\n'
        if not panel:tex+=r'\addlegendentry{Du Fig.15 menu (recomputed)}'+'\n'
        for M in (1,2,4,8):
            tex+=r'\addplot['+colors[M]+r',thick] table[x=Q,y=M'+str(M)+r',col sep=comma]{bb84_comparison.csv};'+'\n'
            if not panel:tex+=r'\addlegendentry{$M\leq '+str(M)+('$ (classical only)' if M==1 else '$ constructions')+'}\n'
            tex+=r'\addplot[forget plot,'+colors[M]+r',only marks,mark=*,mark size=1pt] table[x=Q,y=M'+str(M)+r',col sep=comma]{bb84_anchors.csv};'+'\n'
    tex+=r'''\end{groupplot}
\end{tikzpicture}\par\ref{sharedlegend}\par\vspace{2mm}
\scriptsize Real operations and final $X/Z$ measurements. Lines join certified feasible values.\par
\scriptsize Whole-class certificates are supplied separately at the marked input anchors.
\end{document}
'''
    (fig/'bb84_comparison.tex').write_text(tex,encoding='utf-8')
    atomic_json(fig/'bb84_comparison_provenance.json',dict(points=len(rows),source='bb84_policy_curves_real.json',
       complete_501_grid=len(rows)==501,meaning=src['scope'],
       additional_control='du_Fig15_bb84_recomputed.json; its fixed no-ED policies are included in each budget envelope.',
       anchors='real complete registered-union certificates at Q=.15,.18,.20,.22,.24'))
    print('BB84 figure:',len(rows),'actual certified input points.',flush=True)


if __name__=='__main__':main()
