"""Publish only after the required numerical campaigns complete.

Stops at a rendered PDF; human/model visual inspection and packaging are
separate steps. No preliminary or non-tight value is promoted to an optimum.
"""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];ROOT=PROJECT/'revised'
def read(p):return json.loads(p.read_bytes())
def save(obj):
    p=HERE/'publication_status.json';t=p.with_suffix('.writing.json')
    t.write_text(json.dumps(obj,indent=2)+'\n')
    for attempt in range(100):
        try:t.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)
def run(script,*args):
    env=os.environ.copy();env.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mplconfig'))
    with (HERE/'publication_build.log').open('ab') as log:
        r=subprocess.run([sys.executable,'-X','utf8',str(script),*args],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((str(script),r.returncode))
def replace(path,old,new):
    text=path.read_text(encoding='utf-8')
    if new in text:return
    assert text.count(old)==1,(path,old)
    path.write_text(text.replace(old,new),encoding='utf-8')
def integrate():
    replace(ROOT/'main_text.tex',
        'full compatible-source infimum. Dots are achieved lower values;\nlines guide the eye. Finite-family upper enclosures accompany\nthe BB84 points and selected Werner anchors; other Werner points\ncarry floating-search status. These are not unrestricted capacities\nor guarantees between plotted points.',
        'full compatible-source infimum. Dots are feasible lower values;\nvertical segments give the corresponding finite-family upper bounds.\nAll 120 Werner and 71 BB84 source points have optimization enclosures\nwithin the stated numerical tolerance. The segments are generally\nsmaller than the line width at the displayed rate scale. Connecting\nlines guide the eye; they are not unrestricted capacities or\nguarantees between plotted source errors.')
    replace(ROOT/'sections/s7_certification.tex',
        'The main landscape has 71 BB84 source points and six Werner points\nwith finite-family optimization enclosures. Its additional 114\nWerner points provide achieved rates, rather than upper bounds on\nthe optimum. A plotted lower alone establishes achievability. A\nmemory exclusion requires the complete lower-memory upper bound;\nan optimality claim requires a sufficiently small upper/lower gap.\nThe additional full-Clifford BB84 comparison preserves these distinctions.',
        'The main landscape has 120 Werner and 71 BB84 source points,\neach with enclosures for all eight memory budgets. For Werner, the\nupper--lower gap is at most the larger of $10^{-11}$ bits per raw\npair and $10^{-5}$ times the lower rate. An absolute tolerance does\nnot resolve the relative optimum of an arbitrarily small rate; such\nvalues lie below the displayed range. A plotted lower alone establishes\nachievability. A memory exclusion requires the complete lower-memory\nupper bound. The additional full-Clifford BB84 comparison supplies\ndense feasible curves and separate whole-family bounds at six source\nerrors; it retains any non-tight enclosure explicitly.')
    path=ROOT/'sections/s9_numerical_appendix.tex';text=path.read_text(encoding='utf-8')
    text=text.replace('numerics\\_20261021/','numerics\\_20261022/')
    text=text.replace('Its \\texttt{README.md} gives the stage','Its \\texttt{README.txt} gives the stage')
    path.write_text(text,encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args();prior=None
    while True:
        run(HERE/'collect_revision_results.py');status=read(HERE/'export/status.json')
        # Q=.23 already has a valid freshly rerun complete block upper from
        # 21 October. Its optional further tightening is not needed for the
        # four- and six-slot target exclusions. Do not call it a tight optimum.
        blocks=[str(q) for q in (.19,.2,.21,.22) if not (HERE/'bb84'/str(q)/'block_component/result.json').exists()]
        dense=HERE/'curves_final/status.json'
        dense_complete=dense.exists() and read(dense).get('stage')=='complete'
        key=(status['werner_points'],status['bb84_points'],tuple(blocks),dense_complete)
        save(dict(stage='waiting_numerics',werner=status['werner_points'],bb84=status['bb84_points'],pending_block_sources=blocks,
            pending_dense_update=not dense_complete,missing=status['missing']))
        if key!=prior:print('PUBLICATION WAIT',key,flush=True);prior=key
        if not status['missing'] and not blocks and dense_complete:break
        if not a.watch:return
        time.sleep(60)
    run(HERE/'run_numerical.py','full_witness_profiles.py')
    run(HERE/'run_numerical.py','audit_portfolio_execution.py')
    run(HERE/'audit_exports.py','--require-complete');run(HERE/'collect_revision_results.py','--publish')
    integrate();run(HERE/'make_bb84_table.py')
    run(ROOT/'scripts/make_figures.py');run(ROOT/'build.py','--job','main_updated_20261022')
    run(ROOT/'scripts/check_document.py','--job','main_updated_20261022','--render')
    save(dict(stage='ready_for_visual_review',pdf='revised/main_updated_20261022.pdf',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),document=read(ROOT/'validation/document_checks.json')))
    print('PUBLICATION DATA AND RENDER COMPLETE; VISUAL REVIEW REQUIRED',flush=True)
if __name__=='__main__':main()
