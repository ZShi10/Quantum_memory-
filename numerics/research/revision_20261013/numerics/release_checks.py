"""Manuscript cross-reference/data checks and reproducible PDF review rendering."""
from pathlib import Path
import re
import json
import argparse
import shutil
import subprocess
import hashlib
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]

def check_sources():
    sources=list((ROOT/'paper').glob('*.tex'))+list((ROOT/'paper/sections').glob('*.tex'))
    cited=set()
    for p in sources:
        for match in re.findall(r'\\cite\w*\{([^}]+)\}',p.read_text(encoding='utf-8')):cited.update(match.split(','))
    bib=ROOT/'paper/references.bib';original=bib.read_text(encoding='utf-8')
    entries=re.split(r'(?m)(?=^@)',original);kept=[]
    available=set()
    for entry in entries:
        key=re.match(r'@\w+\{([^,]+),',entry)
        if key and key.group(1) in cited:kept.append(entry.strip());available.add(key.group(1))
    assert available==cited,sorted(cited-available)
    bib.write_text('\n\n'.join(kept)+'\n',encoding='utf-8')
    for ledger in json.loads((ROOT/'validation/claim_ledger.json').read_bytes()):
        for ref in ledger['evidence']:assert (ROOT/ref).exists(),ref
    figuredata=json.loads((ROOT/'validation/figure_provenance.json').read_bytes())
    for f in figuredata:
        p=ROOT/'paper/figures'/(f['figure']+'.pdf')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f['vector_pdf_sha256']
        for ref in f['source_files']:assert (ROOT/ref).exists(),ref
    result=dict(status='passed',used_bibliography_keys=sorted(cited),unused_entries_removed=True,
                major_claim_evidence_links='all exist',figure_hashes='all match')
    (ROOT/'validation/source_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def render():
    exe=shutil.which('pdftoppm')
    if not exe:
        candidate=Path('C:/Users/Frank/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
        if candidate.exists():exe=str(candidate)
    if not exe:raise SystemExit('Poppler pdftoppm required for render review.')
    dst=ROOT/'validation/rendered';dst.mkdir(exist_ok=True)
    reports=[]
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else ImageFont.load_default()
    for stem in ('main','supplement'):
        p=ROOT/'paper'/(stem+'.pdf');reader=PdfReader(str(p));dest=dst/stem;dest.mkdir(exist_ok=True)
        for previous in dest.glob('page-*.png'):
            assert previous.parent.resolve()==dest.resolve()
            previous.unlink()
        r=subprocess.run([exe,'-r','100','-png',str(p),str(dest/'page')],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        pages=[]
        for i,page in enumerate(reader.pages,1):
            text=page.extract_text();assert len(text)>100,(stem,i,'unexpected blank page')
            assert '\ufffd' not in text,(stem,i,'replacement glyph')
            pages.append(dict(page=i,characters=len(text),first_lines=text.splitlines()[:3],last_lines=text.splitlines()[-2:]))
        files=sorted(dest.glob('page-*.png'))
        assert len(files)==len(reader.pages)
        for start in range(0,len(files),6):
            sheet=Image.new('RGB',(1050,980),'#d6dbe0');draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(files[start:start+6]):
                im=Image.open(f);im.thumbnail((336,453));x=(j%3)*350+(350-im.width)//2;y=(j//3)*490+27
                sheet.paste(im,(x,y));draw.text(((j%3)*350+12,(j//3)*490+6),f'{stem} — page {start+j+1}',fill='black',font=font)
            sheet.save(dst/f'{stem}_contact_{start//6+1}.png')
        reports.append(dict(document=stem,pages=pages,pdf_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (ROOT/'validation/pdf_text_render_check.json').write_text(json.dumps(dict(status='passed',reports=reports),indent=2)+'\n')
    print(json.dumps({'status':'rendered','documents':[{r['document']:len(r['pages'])} for r in reports]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');a=p.parse_args()
    render() if a.render else check_sources()
