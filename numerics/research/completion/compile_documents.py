"""TeX Live compilation with binary log capture for mixed Windows encodings."""
from common import *
import subprocess,json,shutil,re,sys
from pypdf import PdfReader


def compile_file(path,destination):
    destination.mkdir(parents=True,exist_ok=True)
    args=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',f'-outdir={destination}',path.name]
    result=subprocess.run(args,cwd=path.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=destination/(path.stem+'.build-output.txt');log.write_bytes(result.stdout)
    assert result.returncode==0,(path,str(log),result.stdout[-2500:].decode('utf-8',errors='replace'))
    pdf=destination/(path.stem+'.pdf');text=(destination/(path.stem+'.log')).read_text(encoding='utf-8',errors='replace')
    bad=[s for s in text.splitlines() if 'Overfull' in s or 'undefined' in s or 'multiply defined' in s or 'Missing character' in s]
    pages=len(PdfReader(pdf).pages)
    print(json.dumps(dict(source=path.name,pdf=str(pdf),pages=pages,warnings=bad)),flush=True)
    return dict(source=str(path),pdf=str(pdf),pages=pages,warnings=bad)


def main():
    project=ROOT.parent.parent;out=project/'output/latex/completion'
    manuscript='main.tex' if '--main' in sys.argv else 'completion_revision.tex'
    results=[compile_file(project/'manuscript'/manuscript,out),compile_file(ROOT/'figures/research_report.tex',out)]
    (DATA/'document_build.json').write_text(json.dumps(dict(results=results),indent=2))


if __name__=='__main__':main()
