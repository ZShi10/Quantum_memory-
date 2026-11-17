"""Certified rectangle unions, without interpolated two-parameter claims."""
from core import *
import csv


def boundary(points):
    edges=sorted({b for b,p in points if b>0})
    if not edges:return [(0.,0.)]
    coords=[(0.,max(p for b,p in points))]
    last=0.
    for edge in edges:
        height=max(p for b,p in points if b>=edge-1e-14)
        coords.extend([(last,height),(edge,height)])
        last=edge
    coords.append((last,0.))
    return coords


def main():
    fig=ROOT/'figures';fig.mkdir(exist_ok=True)
    data=json.loads((DATA/'bb84_asymmetric_grid.json').read_text())['results']
    rows=[];out={}
    for target in (1e-3,1e-5):
        for M in (1,2,4,8):
            points=[]
            for item in data.values():
                if item['results'][f'M{M}']['lower']>=target:
                    points.extend([(item['b'],item['p']),(item['p'],item['b'])])
            coords=boundary(points);out[f'{target:g}_M{M}']=coords
            path=fig/f'bb84_region_{target:g}_M{M}.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['b','p']);writer.writerows(coords)
            rows.append(dict(target_rate=target,M=M,certified_input_points=len(set(points)),boundary_points=len(coords)))
    tex=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=180mm,paperheight=113mm,margin=8mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\pagestyle{empty}
\begin{document}
\centering\small\textbf{BB84 information: guaranteed regions at a nonzero target rate}\par\vspace{2mm}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=15mm},width=76mm,height=72mm,
 xmin=0,xmax=.5,ymin=0,ymax=.5,xlabel={Bit error $b$},ylabel={Phase error $p$},
 xtick={0,.1,.2,.3,.4,.5},ytick={0,.1,.2,.3,.4,.5},grid=major,
 tick label style={font=\scriptsize},label style={font=\small},title style={font=\small},
 legend style={font=\scriptsize,draw=none},legend columns=4]
'''
    colors={1:'cyan!60!black',2:'blue!75!black',4:'orange!85!black',8:'red!75!black'}
    for j,target in enumerate((1e-3,1e-5)):
        tex+=r'\nextgroupplot[title={$R_*='+('10^{-3}' if j==0 else '10^{-5}')+r'$ bit/raw pair},'+('legend to name=regionlegend' if j==0 else 'ylabel={}')+']\n'
        tex+=r'\addplot[gray,densely dashed,domain=0:.5,samples=2] {.5-x};'+'\n'
        for M in (1,2,4,8):
            tex+=r'\addplot['+colors[M]+r',thick] table[x=b,y=p,col sep=comma]{'+f'bb84_region_{target:g}_M{M}.csv'+'};\n'
            if j==0:tex+=r'\addlegendentry{$M\leq '+str(M)+'$}\n'
    tex+=r'''\end{groupplot}
\end{tikzpicture}\par\ref{regionlegend}\par\vspace{1mm}
\scriptsize Real operations and $X/Z$ terminal measurements; optional input Pauli degradation.\par
\scriptsize Each staircase bounds a union of certified rectangles. Dashed line: $b+p=1/2$.
\end{document}
'''
    # Keep the dashed reference out of the memory legend.
    tex=tex.replace(r'\addplot[gray,densely dashed,',r'\addplot[forget plot,gray,densely dashed,')
    (fig/'bb84_regions.tex').write_text(tex,encoding='utf-8')
    atomic_json(fig/'bb84_regions_provenance.json',dict(source='bb84_asymmetric_grid.json',original_points=len(data),
          construction='Mirror certified points by an input Hadamard and take unions of lower-left rectangles using deliberate Pauli degradation. No interpolation of optimal values.',series=rows,boundaries=out))
    print('Certified region figure',len(data),'input points',flush=True)


if __name__=='__main__':main()
