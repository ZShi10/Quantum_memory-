"""Build local delivery archives and smoke-check portable sources.

Only writes inside the explicitly verified research workspace. Existing
source snapshots and earlier manuscript releases are never removed.
"""
from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,zipfile
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
OUT=WORKSPACE/'output/qkd_memory_rewrite_20261014'
STAGE=OUT/'reproducibility'
SKIP_PARTS={'__pycache__','.deps','.runtime','tmp','rendered'}
TEX_AUX={'.aux','.log','.toc','.out','.blg','.fls','.fdb_latexmk'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def run(args,cwd,env=None):
    p=subprocess.run([sys.executable,'-X','utf8',*args],cwd=cwd,env=env,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    if p.returncode:raise RuntimeError(p.stdout[-5000:])
    print('Portable check passed:',args,flush=True)
    return p.stdout
def main():
    OUT.resolve().relative_to(WORKSPACE.resolve())
    assert WORKSPACE.name=='AKD2' and ROOT.name=='revision_20261013'
    OUT.mkdir(exist_ok=True);STAGE.mkdir(exist_ok=True)
    # Source/result snapshots and their textual diagnostics. Rendered page
    # images are regenerated on demand; contact sheets are copied separately.
    for folder in ('paper','data','numerics','reproduction','validation','evidence'):
        for p in sorted((ROOT/folder).rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(ROOT)
            if any(s in SKIP_PARTS for s in rel.parts):continue
            if folder=='paper' and p.suffix in TEX_AUX:continue
            if p.suffix in {'.pyc','.nbi','.nbc'}:continue
            dest=STAGE/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in (ROOT/'validation/rendered').glob('*contact*.png'):
        dest=STAGE/'validation/rendered'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    top=['README.txt','REVIEW_GUIDE_ZH.txt','VALIDATION_REPORT.txt','run.py','build.py','requirements.txt','verify_release.py']
    for name in top:shutil.copy2(ROOT/name,STAGE/name)
    shutil.copy2(ROOT/'inputs/inventory.json',STAGE/'validation/uploaded_input_inventory.json')
    print('Copied portable scientific release',flush=True)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'.runtime')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    tests={}
    # These checks run against the copied package. They do not rely on the
    # original research archive path, graph library, or author input arrays.
    for job in ('controls-check','legacy-audit','literature','results'):
        tests[job]=run(['run.py',job],STAGE,env)
    smoke=ROOT/'tmp/arxiv_source_smoke';(smoke/'paper').mkdir(parents=True,exist_ok=True)
    for p in (STAGE/'paper').rglob('*'):
        if p.is_file() and p.suffix in {'.tex','.bib','.bbl','.pdf'}:
            if p.parent==STAGE/'paper' and p.suffix=='.pdf':continue
            out=smoke/'paper'/p.relative_to(STAGE/'paper');out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    shutil.copy2(STAGE/'build.py',smoke/'build.py');tests['latex']=run(['build.py'],smoke,env)
    pages={}
    for stem in ('main','supplement'):
        a=PdfReader(str(STAGE/'paper'/(stem+'.pdf')));b=PdfReader(str(smoke/'paper'/(stem+'.pdf')))
        assert len(a.pages)==len(b.pages)
        assert [p.extract_text() for p in a.pages]==[p.extract_text() for p in b.pages],stem
        pages[stem]=len(a.pages)
    check=dict(status='passed',portable_jobs=list(tests),latex_extracted_text_identical=True,pages=pages,
               environment='Same tested pinned Python libraries; independent copied source/data paths; clean TeX build',
               scope='Packaging and representative workflow checks, not another whole-class global optimization.')
    dump(STAGE/'validation/package_smoke_check.json',check);dump(ROOT/'validation/package_smoke_check.json',check)
    for name,log in tests.items():
        (STAGE/'validation'/(name+'_portable.log')).write_text(log,encoding='utf-8')
    state=dict(status='completed',request='Execute propmt.txt: scientific reconstruction, figures, numerical evidence, workflow and validation report',
        completed=['New main and nine-section Supplement; four main figures and one diagnostic figure',
                   'Independent graph/code/basis coverage and executable memory schedules',
                   'Fresh matched Werner bounds plus a uniform noise interval',
                   'Six full-z BB84 lower certificates and exact target-memory windows',
                   'Full frozen global-archive integrity and curve-table reconstruction',
                   'Independent Sun reference reconstruction',
                   'PDF compilation, complete-page render review, portable package checks'],
        limitations=['Broad historical global upper optimization remains inherited evidence, explicitly labelled',
                     'No unrestricted LOCC/CRO converse or exhaustive high-N quantum search',
                     'Author metadata and public dataset identifier are left for authors'],
        output_directory=str(OUT),remaining_author_actions=['Review scientific draft','Supply names, affiliations and acknowledgements','Choose a public data repository before submission'])
    dump(ROOT/'STATE.json',state);dump(STAGE/'STATE.json',state)
    files={}
    for p in sorted(STAGE.rglob('*')):
        if p.is_file() and p.name!='release_manifest.json' and '__pycache__' not in p.parts:
            files[p.relative_to(STAGE).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
    manifest=dict(format='qkd-memory-reproducibility-1',files=files,
                  note='Hashes identify the exact release; validate claims using the ledger and explicit protocol scopes.')
    dump(STAGE/'release_manifest.json',manifest)
    run(['verify_release.py'],STAGE)
    # Deliverable PDFs and reading reports also appear at the top level.
    for stem in ('main','supplement'):shutil.copy2(STAGE/'paper'/(stem+'.pdf'),OUT/(stem+'.pdf'))
    for name in ('VALIDATION_REPORT.txt','REVIEW_GUIDE_ZH.txt','README.txt'):shutil.copy2(STAGE/name,OUT/name)
    arxiv=OUT/'arxiv_source.zip'
    with zipfile.ZipFile(arxiv,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted((STAGE/'paper').rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(STAGE/'paper')
            if p.suffix in {'.tex','.bib','.bbl'} or (p.suffix=='.pdf' and rel.parts[0]=='figures'):
                z.write(p,rel.as_posix())
    archive=OUT/'qkd_memory_reproducibility_20261014.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        saved=json.loads(z.read('release_manifest.json'));assert len(saved['files'])==len(files)
        for name,entry in saved['files'].items():
            raw=z.read(name);assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
    with zipfile.ZipFile(arxiv) as z:assert z.testzip() is None
    delivery=dict(status='passed',file_count=len(files),pages=pages,
       artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in
                  (OUT/'main.pdf',OUT/'supplement.pdf',arxiv,archive)},
       complete_evidence_location='reproducibility/evidence/legacy_frozen/ plus reproduction/, validation/, data/',
       inherited_limitation='Historical broad upper optimizations were hash-audited, not independently rerun.')
    dump(OUT/'delivery_manifest.json',delivery);dump(ROOT/'validation/delivery_manifest.json',delivery)
    print(json.dumps(delivery,indent=2),flush=True)
if __name__=='__main__':main()
