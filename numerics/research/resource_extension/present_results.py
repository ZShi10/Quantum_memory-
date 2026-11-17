from common import *
import json,subprocess,shutil,hashlib
import numpy as np

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)


def texnum(x):
    if x<=0:return '--'
    a,b=f'{x:.3e}'.split('e');return rf'${a}\times10^{{{int(b)}}}$'


def main():
    data=json.loads((DATA/'combined_results.json').read_text());six=data['six_state'];bb=data['bb84'];catalog=data['protocols']
    qs=sorted({r['Q'] for r in six});sixmap={(r['Q'],r['M_budget']):r for r in six}
    np.savetxt(FIG/'six.dat',[[q]+[sixmap[q,M]['rate'] or np.nan for M in (1,2,3,4,6,8)] for q in qs],header='Q M1 M2 M3 M4 M6 M8',comments='')
    bs=sorted({r['eb'] for r in bb if r['eb']==r['ep']});bm={(r['eb'],r['ep'],r['permissions'],r['M_budget']):r for r in bb}
    for permission,memories in [('full',(1,2,3,4,6,8)),('real',(2,3,4,6,8))]:
        np.savetxt(FIG/f'bb_{permission}.dat',[[q]+[bm[q,q,permission,M]['rate'] or np.nan for M in memories] for q in bs],header='Q '+' '.join('M'+str(M) for M in memories),comments='')
    sixtable=[];mdsix=[]
    for q in (.20,.23,.28,.30,.305,.31,.315,.32):
        lo=sixmap[q,2];hi=sixmap[q,8];proto=catalog[hi['protocol']]
        sixtable.append(f"{q:.3f} & {texnum(lo['rate'])} & {texnum(hi['rate'])} & {hi['m']} & {hi['M_used']}" + r' \\')
        mdsix.append(f"| {q:.3f} | {lo['rate']:.8e} | {hi['rate']:.8e} | {hi['m']} | {hi['M_used']} | {hi['ED']} | {hi['AD']} |")
    bbtable=[];mdbb=[]
    for q in (.20,.21,.22,.23,.24,.245):
        one,two,eight=[bm[q,q,'full',M] for M in (1,2,8)]
        bbtable.append(f"{q:.3f} & {texnum(one['rate'])} & {texnum(two['rate'])} & {texnum(eight['rate'])} & {eight['M_used']}" + r' \\')
        mdbb.append(f"| {q:.3f} | {one['rate']:.8e} | {two['rate']:.8e} | {eight['rate']:.8e} | {eight['M_used']} | {eight['m_max']} |")
    asym=[]
    for eb,ep in ((.10,.30),(.18,.25),(.12,.35)):
        vals=[bm[eb,ep,'real',2],bm[eb,ep,'real',8],bm[eb,ep,'full',2],bm[eb,ep,'full',8]]
        asym.append(f'{eb:.2f}, {ep:.2f} & '+' & '.join(texnum(v['rate']) for v in vals)+r' \\')
    source=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.55cm]{geometry}
\usepackage{pgfplots,booktabs,amsmath}
\pgfplotsset{compat=1.18,every axis/.append style={grid=major,grid style={gray!15},tick label style={font=\small},label style={font=\small},legend style={font=\small,draw=none},unbounded coords=jump}}
\definecolor{blueone}{RGB}{25,118,163}\definecolor{orangeone}{RGB}{219,122,32}\definecolor{purpleone}{RGB}{119,85,163}\definecolor{greenone}{RGB}{24,132,91}
\setlength{\parindent}{0pt}
\begin{document}
{\Large Expanded resource comparison}\hfill 20 September 2026
\par\medskip
\textbf{One question: which ED and AD design gives a better rate at a stated memory budget?}
All rates are secret bits per original entangled pair. $m$ counts total
inputs to an ED block; $M$ counts peak live qubits \emph{per party}, including
the current input port. Inputs are IID Bell-diagonal; gates and memory
are ideal, and extraction is asymptotic.
\par\medskip
\textbf{Six-state / Werner input.} The envelope includes previous short-block
and restart-pumping witnesses, new graph candidates with $m\leq14$, and
long-code refinements. Every curve is an achieved family bound.
\begin{center}\begin{tikzpicture}
\begin{axis}[width=16cm,height=8.1cm,ymode=log,xmin=.195,xmax=.321,ymin=1e-12,ymax=.04,xlabel={Werner QBER $Q$},ylabel={Secret bits / raw pair},legend columns=3,legend pos=north east]
\addplot[gray,mark=o] table[x=Q,y=M1]{six.dat};\addlegendentry{$M\leq1$}
\addplot[blueone,mark=square] table[x=Q,y=M2]{six.dat};\addlegendentry{$M\leq2$}
\addplot[orangeone,mark=triangle] table[x=Q,y=M3]{six.dat};\addlegendentry{$M\leq3$}
\addplot[greenone,mark=diamond] table[x=Q,y=M4]{six.dat};\addlegendentry{$M\leq4$}
\addplot[purpleone,mark=+] table[x=Q,y=M6]{six.dat};\addlegendentry{$M\leq6$}
\addplot[black,densely dashed,mark=x] table[x=Q,y=M8]{six.dat};\addlegendentry{$M\leq8$}
\end{axis}\end{tikzpicture}\end{center}
\begin{center}\small\begin{tabular}{rrrrr}\toprule
$Q$ & Best found, $M\leq2$ & Best found, $M\leq8$ & Selected $m$ & Implemented $M$\\\midrule
@@SIX@@
\bottomrule\end{tabular}\end{center}
Longer classical codes matter: at $Q=0.28$, a four-input block followed
by two binary AD rounds and an SPC $[10,9,2]$ code reaches
$1.91504\times10^{-4}$. At $Q=0.305$, an eleven-input graph measurement
with an $M=6$ implementation, two AD rounds, and SPC $[29,28,2]$ reaches
$9.19189\times10^{-7}$.
\par\medskip
The larger graph families are sampled constructions. Their quoted memory
is an implementation bound unless it matches a proved lower bound.
The $M=8$ curve may coincide with smaller budgets. A positive-rate
threshold near $1/3$ remains distinct from these finite-noise rate comparisons.
\newpage
{\Large BB84 compatible inputs: full Clifford permissions}
\par\medskip
The input family is $(1-e_b-e_p+z,e_b-z,z,e_p-z)$ with
$0\leq z\leq\min(e_b,e_p)$. No Werner twirl is applied. A legal pilot
test identifies $z$ before a fixed decision rule chooses production ED and
AD. Each plotted rate covers the full compatibility interval.
\begin{center}\begin{tikzpicture}
\begin{axis}[width=16cm,height=8.1cm,ymode=log,xmin=.145,xmax=.247,ymin=1e-11,ymax=.1,xlabel={$e_b=e_p$},ylabel={Guaranteed bits / raw pair},legend columns=3,legend pos=south west]
\addplot[gray,mark=o] table[x=Q,y=M1]{bb_full.dat};\addlegendentry{$M\leq1$}
\addplot[blueone,mark=square] table[x=Q,y=M2]{bb_full.dat};\addlegendentry{$M\leq2$}
\addplot[orangeone,mark=triangle] table[x=Q,y=M3]{bb_full.dat};\addlegendentry{$M\leq3$}
\addplot[greenone,mark=diamond] table[x=Q,y=M4]{bb_full.dat};\addlegendentry{$M\leq4$}
\addplot[purpleone,mark=+] table[x=Q,y=M6]{bb_full.dat};\addlegendentry{$M\leq6$}
\addplot[black,densely dashed,mark=x] table[x=Q,y=M8]{bb_full.dat};\addlegendentry{$M\leq8$}
\end{axis}\end{tikzpicture}\end{center}
\begin{center}\small\begin{tabular}{rrrrr}\toprule
$e_b=e_p$ & $M\leq1$ & $M\leq2$ & $M\leq8$ & Implemented $M$\\\midrule
@@BB@@
\bottomrule\end{tabular}\end{center}
The short-code search includes all code classes through $n=7$ and selected
$n=8$ codes. The comparison also includes direct-measurement and two-register
pumping baselines with binary AD prefixes up to 64 and final repetition
lengths up to 128. These longer baselines are important at high noise:
at $24\%$, the two-register bound is already about
$5.50\times10^{-7}$, compared with about $7.90\times10^{-7}$ for the
selected larger-block rule.
\par\medskip
After measurement-equivalent compilation, the selected larger-block
rules at $23\%$ and $24\%$ need at most three live qubits per party.
The $25\%$ positive-rate supremum is already available without coherent
storage when free $Y$ measurement is allowed. The comparison here concerns
rate and bounded constructed families, not a new unrestricted threshold.
\newpage
{\Large BB84: restricted real operations and protocol records}
\par\medskip
Real operations allow $H$, CNOT and Pauli gates with $X/Z$ measurements.
A two-register Bell-basis pilot can identify $z$. Compilations that use
$S$ gates are available only in the full-Clifford panel on the previous page.
\begin{center}\begin{tikzpicture}
\begin{axis}[width=16cm,height=7.6cm,ymode=log,xmin=.145,xmax=.247,ymin=1e-28,ymax=.1,ytick={1e-2,1e-7,1e-12,1e-17,1e-22,1e-27},xlabel={$e_b=e_p$},ylabel={Guaranteed bits / raw pair},legend columns=3,legend pos=south west]
\addplot[blueone,mark=square] table[x=Q,y=M2]{bb_real.dat};\addlegendentry{$M\leq2$}
\addplot[orangeone,mark=triangle] table[x=Q,y=M3]{bb_real.dat};\addlegendentry{$M\leq3$}
\addplot[greenone,mark=diamond] table[x=Q,y=M4]{bb_real.dat};\addlegendentry{$M\leq4$}
\addplot[purpleone,mark=+] table[x=Q,y=M6]{bb_real.dat};\addlegendentry{$M\leq6$}
\addplot[black,densely dashed,mark=x] table[x=Q,y=M8]{bb_real.dat};\addlegendentry{$M\leq8$}
\end{axis}\end{tikzpicture}\end{center}
\begin{center}\small\begin{tabular}{rrrrr}\toprule
$(e_b,e_p)$ & Real $M\leq2$ & Real $M\leq8$ & Full $M\leq2$ & Full $M\leq8$\\\midrule
@@ASYM@@
\bottomrule\end{tabular}\end{center}
With real permissions, the long-repetition two-register pilot rule also
has a certified positive rate at $e_b=e_p=0.245$: about
$9.43\times10^{-27}$ bits per raw pair. Positive-rate existence and useful
finite-noise yield must therefore be discussed separately.
\par\medskip
\textbf{What each saved protocol specifies.} Input count and scheduling;
gate sequence or graph and local rotations; classical output relabeling;
ED acceptance; binary AD rounds; generator rows or repetition/SPC definition;
and rate normalization including failed attempts. BB84 protocols additionally
include every $z$ cell of the pilot decision rule, rather than only the
circuit at one sampled worst-case input.
\par\medskip
\textbf{Validation.} New graph probabilities were checked against independent
Pauli propagation and exact rational BB84 polynomials. New short-code rates
were checked by an independent Fourier calculation. Twenty selected Werner
witnesses were recomputed at high precision. The new BB84 short-code rules
and long-repetition pilot rules have directed-rounding certificates.
Previously saved real pumping baselines retain their continuous numerical
enclosures; the expanded real long-repetition rules are also certified.
\par\medskip
These plots summarize sampled physical error rates. Lines guide the eye;
they do not certify every switch in the best protocol as $Q$ varies.
No claim is made about arbitrary adaptive circuits, all long codes,
finite-key security, noisy storage, or global fixed-memory optimality.
An omitted positive point is not an impossibility statement.
\end{document}
'''
    source=source.replace('@@SIX@@','\n'.join(sixtable)).replace('@@BB@@','\n'.join(bbtable)).replace('@@ASYM@@','\n'.join(asym))
    (FIG/'resource_tradeoffs.tex').write_text(source,encoding='utf-8')
    report='''# 存储预算与 ED+AD 方案：扩展结果（2026-09-20）

本轮保留原论文不动，将总输入数 m、每方峰值活跃 qubit 数 M、经典码长 n 分开。M 包括输入端口。所有结果为理想 IID Bell 输入、渐近安全熵率，单位是 secret bits/raw pair。

## 先澄清四输入结论

m<=4 的固定 Clifford、一逻辑输出并最终测量的有限类中，full-Clifford 的完整终端测量都能以 M<=2 实现。因此这个子问题的最佳值可用两个活跃 qubit 达到。它没有限制四个 memory 先后处理多少输入，也没有包含任意递归、自适应协议；更不是任意 M<=4 协议的全局最优性证明。等价性针对最终经典数据和 Eve 条件态，不要求逐步重现原线路的中间量子态。

## 扩展了什么

- 5367 个 m=5..14 的图结构/局部换基候选，原始实现预算覆盖 M<=8；另保留此前较小线路、八输入候选和 restart pumping。
- 全部 n<=7 的二进制线性码坐标等价类，以及部分 n=8 中间维数码，共 284 个条目。已知输入或可识别输入的优化利用直和支配性缩减至 69 个计算内核。
- Werner：经典前置 B 轮与长度至 128 的 repetition/SPC 扩展；长码是在短码筛选出的 ED 候选上进一步优化，并非全部 ED 与所有长码的全局搜索。
- BB84：12 个对称/非对称误差点，无 Werner twirl；完整短码反馈规则对全部兼容 z 认证。另加入合法 pilot 后选直接测量或两寄存器 pumping、最多 64 轮 B 和最后长度至 128 的 repetition 基线。
- 对胜出线路继续做 full-Clifford 等价编译，保存实际门、调度和经典输出重标记。real 与 full 权限分别计数；使用 S 的压缩不能送给 real 基线。

## Six-state/Werner 代表协议

下表取所有新旧已找到方案的包络。m 是成功路径或固定 block 的总输入数；pumping 另外用平均重试消耗计率。

| Q | M<=2 最佳已找到率 | M<=8 最佳已找到率 | 后者 m | 后者实现 M | ED 编号 | 后续 AD |
|---|---:|---:|---:|---:|---|---|
'''+ '\n'.join(mdsix)+'''

例如 Q=30.5% 时，十一输入图测量 XG04192 的已找到 full 实现使用 M=6，之后两轮 B 与 [29,28,2] SPC，产率约 9.19e-7；所列 M<=2 家族约 4.14e-7。该测量的下界为 M>=5，尚不能把找到的 M=6 实现称为最小实现，更不能据此证明同样产率至少需要五个 memory。

Q=28% 时，[10,9,2] SPC 将一个 M=2 四输入方案改善到约 1.915e-4。更长码会改善小 memory 的基线，因此扩大 ED 时必须同步扩大经典处理。

## BB84 代表结果（full 权限）

| eb=ep | M<=1 | M<=2 | M<=8 | 后者实现 M | 后者最大生产输入数 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(mdbb)+'''

23% 和 24% 的较大 block 决策规则在等价编译后只需 M<=3。它们按实际可估计的 z 选择不同生产 ED 和 AD，不能用“一个最坏点的线路”替代完整协议。尤其在 24%，补上长 repetition 基线后，M<=2 已能达到约 5.50e-7，与较大 block 规则约 7.90e-7 比较更为公平。

BB84 的 full 权限允许 Y 测量；其无相干存储正率边界已能逼近 25%。论文应强调受限权限下的阈值构造，以及同等权限、明确计算范围内的产率改善。

real 权限的 M=2 基线也补做了合法 pilot 后选 pumping 深度和长 repetition 的完整规则。eb=ep=24.5% 时，得到约 9.43e-27 bits/raw pair 的严格正率下界；这说明正率边界与有限噪声下的产率必须分开讨论，不能把扫描中的数值零当作不可能性界。

## 文章主线及交付文件

主要结论应围绕“明确存储预算下 ED 与 AD 的联合设计和完整密钥率”展开。两寄存器阈值构造、memory 编译、fidelity 反例、单轮码反例分别为这条主线服务，不并列写成互不相关的发现。具体组织见 ARTICLE_DIRECTION.md。

- data/combined_results.json：所有预算和扫描点的胜出协议，以及 BB84 完整 z 区间规则；不是只给出 key-rate 数字。
- PROTOCOL_CATALOG.md、data/protocol_catalog.json：具体图边、逻辑算符、门与调度、生成矩阵及明确保留的误差陪集。反馈区间使用 t=z/min(eb,ep)，并非 z 本身。
- data/six_state_summary.csv、data/bb84_summary.csv：便于比较的参数表。
- data/memory_refinement.json：实际编译、输入调度、经典重标记；小线路 full 实现与较大图的局部互补优化分别保存。
- data/witnesses_130digit.json、bb84_certified_*.json、bb84_long_repetition_M*_certified.json：高精度或有向舍入证书。
- data/validation_summary.json：本轮汇总检查。

这里给出的 Q 或 (eb,ep) 是已扫描点，连续物理误差区间中协议切换的精确边界尚未认证。较大 ED、n=8 码和长码组合仍是构造/筛选搜索；完整枚举、最佳已找到值、严格存在性定理须分别陈述。一般 CRO 上限和无限协议的固定-memory 全局最优性按约定不纳入本轮结论。
'''
    (ROOT/'RESULTS.md').write_text(report,encoding='utf-8')
    compiler=shutil.which('pdflatex');assert compiler;out=ROOT.parents[1]/'output/pdf'
    run=subprocess.run([compiler,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={out.as_posix()}','resource_tradeoffs.tex'],cwd=FIG,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (FIG/'compile_console.log').write_bytes(run.stdout)
    if run.returncode:print(run.stdout.decode('utf-8',errors='replace')[-3500:]);raise SystemExit(run.returncode)
    log=(out/'resource_tradeoffs.log').read_text(encoding='utf-8',errors='replace');assert 'Overfull' not in log
    print('Compiled',out/'resource_tradeoffs.pdf')


if __name__=='__main__':main()
