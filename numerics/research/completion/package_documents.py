"""Publish the compiled, visually reviewed research documents within the workspace."""
from common import ROOT, DATA
import hashlib
import json
import shutil
from pathlib import Path
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = ROOT.parent.parent
    build = json.loads((DATA / 'document_build.json').read_text())
    destinations = {'main.tex': 'main_revised.pdf',
                    'research_report.tex': 'completion_research_report.pdf'}
    target_dir = project / 'output' / 'pdf'
    target_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    for record in build['results']:
        source, built = Path(record['source']), Path(record['pdf'])
        assert not record['warnings'], record
        assert source.name in destinations, source
        pages = len(PdfReader(built).pages)
        assert pages == record['pages']
        target = target_dir / destinations[source.name]
        shutil.copy2(built, target)
        assert digest(target) == digest(built)
        documents.append(dict(
            source=str(source.relative_to(project)).replace('\\', '/'),
            output=str(target.relative_to(project)).replace('\\', '/'),
            pages=pages, source_sha256=digest(source),
            output_sha256=digest(target)))
    assert len(documents) == 2
    result = dict(status='passed', documents=documents,
                  visual_review='All final pages reviewed before packaging; renders in output/qa/completion/final.')
    (DATA / 'delivery.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
