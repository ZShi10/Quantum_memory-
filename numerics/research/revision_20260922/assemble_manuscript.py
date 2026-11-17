"""Assemble the new manuscript from verified sections and generated artifacts.

Default destination is a working draft. --final requires completed primary
data and preserves the previous manuscript in a dated history directory.
"""
from core import *
import argparse,shutil,subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');a=parser.parse_args()
    draft=ROOT/'draft'
    if a.final:
        subprocess.run([sys.executable,str(ROOT/'audit_final_artifacts.py'),'--require-complete'],check=True)
        union=json.loads((DATA/'joint_union_summary.json').read_text())
        assert union['complete_Werner_anchors']==6,'Werner union certificate still incomplete'
        bb=json.loads((DATA/'bb84_policy_curves_real.json').read_text())
        assert len(bb['results'])>=501,'BB84 curve still incomplete'
        for q in (.15,.18,.2,.22,.24):
            src=json.loads((DATA/f'bb84_joint_extension_real_{q:.5f}_{q:.5f}_n128_menu_certified.json').read_text())
            assert src['complete']
        destination=PROJECT/'manuscript'
        history=destination/'history/before_joint_revision_20260926';history.mkdir(parents=True,exist_ok=True)
        for name in ('main.tex','references.bib'):
            if not (history/name).exists():shutil.copy2(destination/name,history/name)
    else:destination=ROOT/'paper_draft'
    (destination/'sections').mkdir(parents=True,exist_ok=True);(destination/'figures').mkdir(exist_ok=True)
    for file in draft.glob('*.tex'):shutil.copy2(file,destination/'sections'/file.name)
    foundations=(draft/'foundations.tex').read_text(encoding='utf-8')
    marker=r'\subsection{Quantum blocks, pumping and recursion}'
    foundations=foundations.replace(marker,r'\input{sections/protocol_figure}'+'\n\n'+marker)
    (destination/'sections/foundations.tex').write_text(foundations,encoding='utf-8')
    base=(draft/'code_tails_base.tex').read_text(encoding='utf-8')
    extra=r'''
For compressed OTP a code of dimension $k$ satisfies
\begin{equation}\label{eq:comp-tail}
 r_{\rm comp}(\pp)\leq1-H(\pp)+\frac{k}{n}.
\end{equation}
Indeed, $[n-H(X^n,Z^n|s)]_+\leq n-H(Z^n|X^n,s)$, since the latter is nonnegative. Averaging gives a gross rate at most $1-H(Z|X)$. Also $H(S)=nH(X)-H(X^n|S)\geq nH(X)-k$. Subtracting proves the bound. For Eq.~\eqref{eq:codewitness}, $H(\pp)>1.11963$, so compressed-OTP repetition has nonpositive net rate for all $n\geq9$. Plain OTP is dominated by Proposition~\ref{prop:plainotp}. The SPC conditional-entropy bound above exceeds one for both parity flags at every $n\geq64$, so its gross OTP rate is zero as well. Outward evaluation through length 128, including the Watanabe terminal, and the analytic tails therefore prove Proposition~\ref{prop:code} for all three modes. No B prefix is included in that proposition.
'''
    base=base.replace('For the witnesses,','For the fixed input in Eq.~\\eqref{eq:codewitness},')
    (destination/'sections/code_tails.tex').write_text(base+extra,encoding='utf-8')
    for name in ('werner_comparison','bb84_comparison','bb84_regions'):
        pdf=ROOT/f'figures/build/{name}.pdf'
        if pdf.exists():shutil.copy2(pdf,destination/f'figures/{name}.pdf')
    # The working PDF may be compiled before the last dense curve finishes;
    # its source is never copied to the final destination until the checks above pass.
    references=(PROJECT/'manuscript/references.bib').read_text(encoding='utf-8')
    additions=(draft/'additional_references.bib').read_text(encoding='utf-8')
    if '@article{Watanabe,' not in references:references+='\n'+additions
    (destination/'references.bib').write_text(references,encoding='utf-8')
    text=r'''% !TEX program = pdflatex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,amssymb,amsthm,mathtools}
\usepackage{graphicx,booktabs,array,longtable,placeins}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,allcolors=blue!50!black]{hyperref}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\theoremstyle{definition}\newtheorem{definition}[theorem]{Definition}
\newcommand{\pp}{\boldsymbol p}
\newcommand{\pos}[1]{[#1]_+}
\newcommand{\Ftwo}{\mathbb F_2}
\DeclareMathOperator{\rank}{rank}
\title{Quantum key distribution with bounded quantum memory:\\joint entanglement and advantage distillation}
\author{}
\date{}
\begin{document}
\maketitle
\begin{abstract}
Quantum memory can improve both the noise tolerance and the yield of quantum key distribution. We study their dependence on the peak live local register for ideal, independent Bell-diagonal inputs. An exact streaming construction counts the memory of complete stabilizer measurements, distinguishing simultaneous storage from total input count. Two live qubits per party suffice for positive rate below the Werner boundary $Q=1/3$ and, without Werner twirling, uniformly over the BB84-compatible family with bit and phase error sum below $1/2$. We then jointly optimize quantum circuits, classical linear codes and paid one-time-pad processing within declared finite implementation classes, and certify rate intervals at selected inputs. Larger registers improve nonzero rates substantially: at Werner $Q=0.30$, two- and seven-register constructions yield approximately $2.93\times10^{-6}$ and $2.28\times10^{-4}$ bits per raw pair. For BB84 error information with $b=p=0.22$ and $X/Z$ measurements, two and five registers yield approximately $7.71\times10^{-4}$ and $1.73\times10^{-3}$. These comparisons separate memory benefits from measurement permissions and account for all failed attempts and consumed secret-key resources.
\end{abstract}
\input{sections/foundations}
\input{sections/memory_thresholds}
\input{sections/finite_class}
\input{sections/results}
\appendix
\input{sections/memory_threshold_proofs}
\input{sections/security}
\input{sections/code_tails}
\input{sections/region_lemma}
\input{sections/decoder_gates}
\input{sections/enumeration}
\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
'''
    (destination/'main.tex').write_text(text,encoding='utf-8')
    print(destination/'main.tex',flush=True)


if __name__=='__main__':main()
