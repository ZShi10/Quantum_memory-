"""Provenance and cross-artifact checks; no numerical search is rerun."""
from common import *
from fractions import Fraction as F
from decimal import Decimal as D
from datetime import datetime,timezone
import ast,hashlib,json,platform
from pypdf import PdfReader


def read(path):return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_policy(policy,lower):
    end=F(0);minimum=D('Infinity')
    for cell in policy:
        a,b=map(F,cell['interval']);assert a==end and b>a;end=b
        minimum=min(minimum,D(str(cell['lower'])))
    assert end==1 and D(str(lower))<=minimum
    return len(policy)


def main():
    workspace=ROOT.parents[1]
    protected={'manuscript/main.tex':'f5373add7c7accbef6c5d625ff599b1ba26b8fc3bd0b4906b3e028725ec1f689',
               'output/pdf/main.pdf':'3802f6c34064188dd155c3cf0303b165ded50f6e45763c64b085550b5987da26'}
    for path,value in protected.items():assert digest(workspace/path)==value,path
    short_rules=short_cells=long_rules=long_cells=0
    for path in DATA.glob('bb84_certified_[0-9]*.json'):
        src=read(path);assert src['status']=='passed'
        assert digest(DATA/src['source'])==src['source_sha256']
        for r in src['results'].values():
            if D(r['lower'])<=0:continue
            short_rules+=1;short_cells+=check_policy(r['policy'],r['lower'])
    for path in DATA.glob('bb84_long_repetition_M*_certified.json'):
        src=read(path);assert src['status']=='passed'
        for r in src['results']:
            if D(r['lower'])<=0:continue
            long_rules+=1;long_cells+=check_policy(r['policy'],r['lower'])
            if src.get('permissions')=='real':
                assert all(c['parameters']['family']=='pumping' or c['parameters']['axis'] in (0,1) for c in r['policy'])
    six=read(DATA/'six_state.json');signatures={}
    for item in six['results']:
        for r in item['results'].values():
            assert all(b['contribution']>0 or b.get('code') is None for b in r['branches'])
            sig=json.dumps([{k:v for k,v in b.items() if k!='contribution'} for b in r['branches'] if b['contribution']>0],sort_keys=True)
            key=(r['Q'],r['ED']);assert key not in signatures or signatures[key]==sig
            signatures[key]=sig
    for r in read(DATA/'witnesses_130digit.json')['results']:
        sig=json.dumps([{k:v for k,v in b.items() if k not in ('contribution','rate_130digit')} for b in r['branches']],sort_keys=True)
        assert signatures[r['Q'],r['ED']]==sig
    catalog=read(DATA/'protocol_catalog.json');source_files=set()
    for pid,r in catalog['protocols'].items():
        path=workspace/r['source_file'];assert path.exists();source_files.add(path)
        eds=r.get('EDs',[r.get('ED')]);required=[]
        for ed in eds:
            e=catalog['ed_definitions'][ed]
            if 'full_terminal_implementation' in e:
                impl=e['full_terminal_implementation']
                cost=impl.get('M_full',impl.get('M')) if r.get('permission','full')=='full' else e['M']
            else:cost=e.get('M',2 if ed.startswith(('pump_','restart_pump_')) else 1)
            required.append(cost)
        if r.get('permission')=='real':required.append(2)
        assert max(required)<=r['M']<=r['budget'],(pid,max(required),r['M'])
        for cell in r.get('policy',[]):
            for b in cell.get('processing',[]):assert b['code'] in catalog['code_definitions']
    pdf=workspace/'output/pdf/resource_tradeoffs.pdf';reader=PdfReader(pdf)
    assert len(reader.pages)==3
    assert all((p.extract_text() or '').strip() for p in reader.pages)
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    for name in ('character_verification.json','rate_verification.json','witnesses_130digit.json','memory_refinement.json','validation_summary.json'):
        assert read(DATA/name)['status']=='passed'
    result=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        original_manuscript_and_pdf_unchanged=protected,protocol_records=len(catalog['protocols']),
        explicit_ED_definitions=len(catalog['ed_definitions']),explicit_code_definitions=len(catalog['code_definitions']),
        positive_short_BB84_rules=short_rules,short_BB84_certified_cells=short_cells,
        positive_long_BB84_rules=long_rules,long_BB84_certified_cells=long_cells,
        pdf_pages=len(reader.pages),scope='Cross-file consistency, source hashes, memory feasibility, exact policy coverage and preserved manuscript. This audit does not prove global optimality or replace the numerical verification/certificate calculations.')
    (DATA/'final_audit.json').write_text(json.dumps(result,indent=2))
    files={p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.npz','.csv','.tex','.dat') and p.name!='artifact_manifest.json'}
    files.update(OLD.glob('*.py'));files.update(source_files)
    files.update(OLD/'data'/n for n in ('blocks.json','codes.json','graph_family_blocks.json','graph_measurements.json'))
    files.update((workspace/'manuscript/code').glob('*.py'));files.add(pdf)
    manifest={p.relative_to(workspace).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
    (DATA/'artifact_manifest.json').write_text(json.dumps(dict(utc=result['utc'],files=manifest),indent=2))
    print(json.dumps(result,indent=2));print('Manifest files',len(manifest))


if __name__=='__main__':main()
