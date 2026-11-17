"""Package the visually reviewed 2026-09-26 manuscript without deleting history."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PAPER = PROJECT / 'manuscript'
OUTPUT = PROJECT / 'output'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run([sys.executable, str(ROOT / 'audit_final_artifacts.py'),
                    '--require-complete'], check=True)
    audit = json.loads((ROOT / 'data/final_artifact_audit.json').read_text())
    assert audit['status'] == 'passed'
    pdf = PAPER / 'build/main.pdf'
    log = (PAPER / 'build/main.log').read_text(encoding='utf-8', errors='replace')
    warnings = re.findall(r'^.*(?:LaTeX Warning:|Package .+ Warning:|Overfull \\[hv]box|undefined references).*$',
                          log, re.MULTILINE)
    assert not warnings, warnings
    info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True,
                          text=True, encoding='utf-8', errors='replace').stdout
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE).group(1))
    assert pages == 24

    # Follow the actual TeX input closure; omit historical and unused sections.
    sources = {PAPER / 'main.tex'}
    pending = list(sources)
    while pending:
        current = pending.pop()
        for name in re.findall(r'\\input\{([^}]+)\}', current.read_text(encoding='utf-8')):
            child = (PAPER / name).with_suffix('.tex')
            assert child.is_file() and child.resolve().is_relative_to(PAPER.resolve())
            if child not in sources:
                sources.add(child)
                pending.append(child)
    sources.add(PAPER / 'references.bib')
    for name in ('werner_comparison', 'bb84_comparison', 'bb84_regions'):
        figure = PAPER / f'figures/{name}.pdf'
        assert sha(figure) == sha(ROOT / f'figures/build/{name}.pdf')
        sources.add(figure)
    assert all(p.stat().st_mtime <= pdf.stat().st_mtime for p in sources)

    history = PAPER / 'history/before_joint_revision_20260926'
    history.mkdir(parents=True, exist_ok=True)
    readme = PAPER / 'README.md'
    if readme.exists() and not (history / 'README.md').exists():
        shutil.copy2(readme, history / 'README.md')
    readme.write_text('''# Bounded-memory ED+AD manuscript — 2026-09-26

This is the completed revision within the agreed finite protocol classes and the ideal IID Bell-diagonal model. The compiled paper has 24 pages. The current delivery is `../output/pdf/main_revised.pdf`.

Compile in this directory (TeX Live with TikZ/PGFPlots, natbib and Latin Modern):

```text
mkdir build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

If `build` already exists, omit `mkdir build`. All inputs and the three external vector figures are included in the source archive; the protocol diagram is generated in TeX. Author names and affiliations have intentionally been left blank.

The current numerical entry point is `../research/revision_20260922/REPRODUCIBILITY.md`. The protocol registry, saved witnesses, interval certificates, figure provenance and source hashes are in that research directory. Its frozen dependencies in `research/completion`, `research/joint_optimization` and `research/resource_extension` must be retained when moving the research project. The manuscript-only source archive is sufficient to compile the paper, but is not a replacement for the full numerical project.

The historical `manuscript/code`, `manuscript/data`, `source_snapshot` and `completion_revision.tex` are not the simulation entry points for this version. The preceding main source and README are preserved under `history/before_joint_revision_20260926`.

The paper separates constructive rate curves from whole-class certificates at selected inputs. Its two-qubit positivity theorem holds strictly below the stated boundaries; it is not a positive-rate claim at a separable boundary. General CRO impossibility bounds and unrestricted fixed-memory optimization remain outside this revision.
''', encoding='utf-8')
    sources.add(readme)

    target = OUTPUT / 'pdf/main_revised.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = OUTPUT / 'pdf/history/before_joint_revision_20260926/main_revised.pdf'
    if target.exists() and not previous.exists() and sha(target) != sha(pdf):
        previous.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, previous)
    shutil.copy2(pdf, target)
    assert sha(target) == sha(pdf)

    report = OUTPUT / 'REVISION_REPORT_20260926.md'
    report.write_text('''# 论文修订交付说明（2026-09-26）

本轮约定范围内的研究和全文修订已经完成。正式 PDF 为 `output/pdf/main_revised.pdf`，共 24 页；英文源文件为 `manuscript/main.tex`。旧正式稿与旧 PDF 均已另行备份。全文以“可执行的 memory 计数—少量 memory 的正率边界—有限协议类内的 ED+AD 联合优化”为主线，性能改进并入联合优化结果。

## 核心结果

1. **明确资源和协议。** M 是每方峰值活跃 qubit 数，包含新输入端口；m 是固定量子块的输入数；C 是计入全部失败与重试的平均原始对消耗。给出四种两输入 decoder 的门序列、raw pumping 与递归的重试规则，以及末端测量可转为经典处理时的计数方法。完整 stabilizer 终端测量有精确的顺序实现 memory 公式；四输入以内的 2448 个基均可用至多两个活跃 qubit 实现。这不等于保留任意 ED 量子输出时也只需两个 qubit。
2. **两 qubit 的统一正率构造。** 对 Werner 的 Q<1/3，以及仅已知 bit/phase error 的 BB84 家族 b+p<1/2，给出有限 pumping 长度和有限 AD 长度，使每方两个活跃 qubit 即可产生正率。BB84 的同一协议覆盖所有兼容的隐藏相关参数，不作 Werner twirl。边界是上确界，边界点本身不宣称有正率。这个结果与数值搜索的长度截断分开。
3. **额外 memory 提高非零密钥率。** 在 Werner Q=0.30 时，M≤2 的构造约为 2.92524×10⁻⁶ bits/raw pair，M≤7 约为 2.28404×10⁻⁴。BB84 主比较固定 real 操作与 X/Z 测量；在 b=p=0.22 时，M≤2 约为 7.71319×10⁻⁴，M≤5 约为 1.72772×10⁻³。论文另列目标率 10⁻² 至 10⁻⁶ 下实际达到的最大已计算误差点，突出有用的非零率区间。
4. **联合优化确实必要。** 同一三输入搜索类、相同 AD 菜单下，最大 fidelity 的输出会漏掉更高最终 key rate；有严格控制上下界。固定另一条 ED 线路时，compressed OTP 带来约 6.25% 的提升。证明同一码的 plain OTP 被 noOTP 支配，并给出 compressed OTP 的收益上界。
5. **修正 endpoint-code 猜想。** 给出一个 ED 可达的有理 Bell 输入：单轮 [6,4,2] noOTP 率大于 0.01439497134150，而全部长度 repetition/SPC、三种 Du 模式及 Watanabe 单轮终端的率小于 0.01383047967224。无限码长由解析尾界覆盖。该反例针对固定 AD 输入、单轮比较；加入前置 B 轮后，endpoint 码可以反超，因此没有把它扩张成任意处理下的普遍优越性结论。

## 数值覆盖与图表

- 完整小块类：Werner 输入 m≤8，BB84 输入 m≤4；较大构造覆盖四 decoder 的全部深度≤7 同质递归和常 decoder 泵浦长度≤160。它们不是全部大尺寸 Clifford 电路。
- 经典菜单：B≤16，全部 n≤7 二进制线性码及长度≤128 的 repetition/SPC，三种 Du 通信模式和 Watanabe 终端。BB84 使用事先固定的 ED、B 前缀及码/模式配比，对连续隐藏相关参数取最坏值。
- 主结果有 Werner 六个、BB84 五个噪声锚点，分别覆盖 M=1…8，共 88 组有限类并集上下界。极小率处按实际绝对 gap 报告，不称高相对精度最优。
- Werner 主曲线含 480 个实际共同网格点，BB84 主曲线含 501 点。BB84 正率点均具有完整隐藏参数区间下界；这些密集曲线是已保存构造的可达包络，不是每一点的完整类最优性证书。
- 非对称 BB84 在 182 个半平面输入点计算并作 H 对称扩展。局部 Pauli 加噪引理将各点保证扩展为左下矩形，得到连续可达区域；没有用平滑插值代替证明。
- 另行给出允许 Y 测量的权限对照。无 memory 时改变测量权限本身可能改变阈值，因而没有把它记作 memory 的收益。
- 保存了 629 组相邻已计算点的协议选择记录和 283 个协议定义。这些区间只是网格记录的分组，不是连续噪声区间上的最优性证明。

## 前人工作的比较

Sun 原作者代码固定到明确提交，八条可取得的 500 点原数组与独立复算逐点一致；第九条按同一公式重建。图中分开标识原文熵口径和采用共同 AD 菜单的强化控制。

Du Fig.15 的 noOTP repetition/SPC（长度 2…8）与 Watanabe 对照已按公开公式重算，分别绘入 Werner 和 BB84 图；没有取得作者原数组，所以标记为 recomputed，而不是宣称原始数据逐点复现。对另一些高产率的自适应 ED 文献也给出独立复算和资源口径说明，没有声称本文在所有设置下优于已有工作。

## 核查与复现

四 decoder 的 Pauli 传播、带符号的 measurement compiler、Du 的 cq 熵恒等式、SPC 求值、区间导数、终端经典化恒等式及 OTP 支配关系均有独立检查。最终跨文件核查 44 项通过；正式 PDF 已编译并逐页检查，引用、公式和图表无未解决的编译问题。

复现入口：`research/revision_20260922/REPRODUCIBILITY.md`；协议范围：`PROTOCOL_REGISTRY.md`；主表：`data/table_joint_union.csv`；详细协议：`representative_protocols.json`、`fixed_block_protocols.json`；完整审计：`data/final_artifact_audit.json`。`output/manuscript_source_20260926.zip` 是可独立编译的论文源文件包；数值复现仍使用工作区的完整 research 目录。

## 结论适用范围及后续

本稿研究理想门与存储、IID Bell-diagonal 输入下的渐近净 bits/raw pair。正文给出这一模型下的安全论证、明确的有限类及对应证据，不将结果表述为实验 bit/s 或有限密钥结果。一般 CRO 的 20%/27.6% 不可能性证明、无限协议类下固定 memory 的全局最优性，按约定留待后续；退相干、有限等待时间和有限密钥也未作为本轮已完成结果。投稿时需补入真实作者与单位，并按目标期刊调整版式。
''', encoding='utf-8')

    progress = ROOT / 'PROGRESS.md'
    old_progress = progress.read_text(encoding='utf-8')
    marker = '\n## 历史进度记录（以下不代表当前状态）\n'
    history_text = old_progress.split(marker, 1)[1] if marker in old_progress else old_progress
    latest = '''# 可续接进度

最后更新：2026-09-26。本轮约定的有限协议类研究与全文修订已完成，正式 PDF 已交付。以下完成状态优先于全部历史记录。

## 当前完成状态

- 正式英文稿 `manuscript/main.tex` 已完整替换；`output/pdf/main_revised.pdf` 为本轮 24 页新版。旧 TeX/README 与旧 PDF 均保存在各自的 `history/before_joint_revision_20260926`。
- Werner 六锚点的完整 m≤8 核心及递归/泵浦并集认证完成；BB84 real 五锚点的完整 m≤4 核心及递归/泵浦并集认证完成。共 88 组 M=1…8 上下界。
- BB84 full（额外 Y 权限）四锚点补充对照全部结束，Q=.22/.24 使用较松的约 2e-6 绝对精度，表中保留实际 gap；不能套用 real 主结果的精度或完整块并集范围。
- Werner 主曲线 480 个实际点、BB84 主曲线 501 点均完成。Du 两模型对照重算完成并入图；BB84 与主曲线使用同一精确网格。非对称半平面 182 点、连续可达矩形、40 组目标率记录及协议分组已导出。
- 两寄存器统一正率、精确 stabilizer 终端 memory、连续 BB84 固定协议安全、OTP 支配、零坐标/直和约简、单轮中间维数码的全部 endpoint 长度反例、Pauli 加噪区域扩展均已写入正文或证明附录。
- Q=.28 的 LC4_005_v9 长码见证与旧短码记录的关联已补独立认证；所有导出完整块见证有匹配证书。终端等价、SPC 快速界、零坐标约简均完成独立验证。
- `data/final_artifact_audit.json`：44 项通过。正式 PDF 编译无未解决警告，24 页已逐页视觉核查。交付哈希与文件清单见 `output/delivery_manifest_20260926.json`。
- 中文交付说明：`output/REVISION_REPORT_20260926.md`。当前复现说明：`REPRODUCIBILITY.md`。可独立编译源包：`output/manuscript_source_20260926.zip`。旧表中的后台会话均已结束，不应重复续跑它们。

本轮无待完成的计算作业。后续若继续研究，应从明确新增的范围开始；一般 CRO converse 和无限协议全局最优性仍按用户约定延期。不要把密集可达曲线称为逐点完整类最优，不要把边界上确界称为边界正率，也不要把 `full` 文件名解释为任意大 Clifford 递归。
'''
    progress.write_text(latest + marker + history_text, encoding='utf-8')
    repro = ROOT / 'REPRODUCIBILITY.md'
    text = repro.read_text(encoding='utf-8')
    text = text.replace('当前完成状态以 [PROGRESS.md](PROGRESS.md) 最上方和 `data/final_artifact_audit.json` 为准。正式交付前，工作稿位于 `paper_draft`；不能把旧 `manuscript/main.tex` 当作本轮完成稿。',
                        '本轮已完成，正式稿为 `manuscript/main.tex`，交付 PDF 为 `output/pdf/main_revised.pdf`（24 页）。当前状态与校验记录见 [PROGRESS.md](PROGRESS.md) 最上方、`data/final_artifact_audit.json` 和 `output/delivery_manifest_20260926.json`。`paper_draft` 是此前工作稿，当前应以正式稿为准。')
    repro.write_text(text, encoding='utf-8')

    archive = OUTPUT / 'manuscript_source_20260926.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(sources):
            bundle.write(source, source.relative_to(PAPER).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {p.relative_to(PAPER).as_posix() for p in sources}

    release = dict(
        status='complete_within_declared_scope',
        created_at=datetime.now().astimezone().isoformat(),
        pdf=str(target.relative_to(PROJECT)), pdf_sha256=sha(target), pages=pages,
        source_archive=str(archive.relative_to(PROJECT)), source_archive_sha256=sha(archive),
        source_files_sha256={p.relative_to(PAPER).as_posix(): sha(p) for p in sorted(sources)},
        artifact_audit_sha256=sha(ROOT / 'data/final_artifact_audit.json'),
        artifact_audit_passed=len(audit['checks']),
        compiler_warnings=warnings,
        visual_review=dict(pages=list(range(1, pages+1)),
                           method='Poppler renders and six contact sheets; equations, tables, figures and references checked'),
        scientific_scope='Ideal IID Bell-diagonal model; stated finite classes and analytic positivity theorem. Dense achievable curves are separate from anchor optimality certificates.',
        report=str(report.relative_to(PROJECT)))
    (OUTPUT / 'delivery_manifest_20260926.json').write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(pdf=str(target), pages=pages, pdf_sha256=sha(target),
                          source_files=len(sources), archive=str(archive)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
