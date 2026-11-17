"""Cross-artifact audit of the declared completed research scope."""
from common import *
import ast,json,hashlib,sys,platform,csv
from decimal import Decimal as D
from fractions import Fraction as F
from collections import Counter
import numpy as np


def read(p):return json.loads(p.read_text())
def main():
    result=read(DATA/'current_results.json');catalog=read(DATA/'protocol_catalog.json');cert=read(DATA/'envelope_werner_certificates.json')['results']
    assert len(result['werner_complete_core'])==84 and len(result['bb84_complete_core'])==36
    assert len(read(DATA/'werner_complete_k16.json')['results'])==21
    assert sum(len(read(f)['results']) for f in DATA.glob('werner_complete_long*.json'))==21
    assert len(read(DATA/'larger_werner.json')['results'])==21
    assert len(read(DATA/'pumping_fair_codes.json')['results'])==21
    assert len(read(DATA/'werner_complete_library.json')['entries'])==6895
    for r in result['werner_complete_core']:assert 0<=r['lower']<=r['upper']
    for r in result['bb84_complete_core']:assert 0<=r['lower']<=r['certified_upper']
    w={(r['Q'],r['budget']):r for r in result['six_state']}
    bb={(r['eb'],r['ep'],r['permissions'],r['budget']):r for r in result['bb84']}
    for (q,M),r in w.items():
        assert r['memory']<=M
        if M>1:assert r['rate']+max(1e-25,r['rate']*2e-10)>=w[q,M-1]['rate']
        p=result['protocols'][r['protocol']]
        if r['rate']>0:
            assert p['certificate_id'] in cert
            c=cert[p['certificate_id']]
            assert D.from_float(r['rate'])<=D(c['lower'])<=D(c['upper'])
    for (b,p,gate,M),r in bb.items():
        assert r['memory']<=M and r['lower']>=0
        if (b,p,gate,M-1) in bb:assert r['lower']+max(1e-40,r['lower']*1e-11)>=bb[b,p,gate,M-1]['lower']
        if gate=='full' and (b,p,'real',M) in bb:assert r['lower']+max(1e-40,r['lower']*1e-11)>=bb[b,p,'real',M]['lower']
        rec=result['protocols'][r['protocol']]
        assert 'complete_' in rec['source'] or rec.get('certificate') in ('60-digit outward rounding','fixed_pumping_certified.json')
    for pid,rec in result['protocols'].items():
        ids=rec.get('EDs',[rec['ED']] if 'ED' in rec else [])
        for eid in ids:
            assert eid in catalog['ed_definitions'],(pid,eid)
            e=catalog['ed_definitions'][eid]
            if 'streaming' in e:assert e['streaming']['M']<=rec['budget']
    # Independent policy-domain coverage, including source/certificate agreement.
    policy_cells=0
    for path in DATA.glob('bb84_complete_*_certified.json'):
        c=read(path);src=read(path.with_name(path.name.replace('_certified','')))
        for mode,rule in src['results'].items():
            end=F(0)
            for cell in rule['policy']:
                a,b=map(F,cell['interval']);assert a==end and b>a;end=b;policy_cells+=1
                assert cell['ED'] in catalog['ed_definitions'] or cell['ED'].startswith('E')
            assert end==1
            cc=c['results'][mode];assert D(cc['lower'])==min(D(cell['lower']) for cell in cc['cells'])
    checks=['small_memory_exact.json','graph_memory_exact.json','memory_eight_graphs.json','float_enclosure_validation.json','spc_dynamic_validation.json','spc_upper_validation.json','real_class_coverage.json','fixed_pumping_certified.json']
    for f in checks:assert read(DATA/f)['status']=='passed',f
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    count=Counter(e['M'] for e in read(DATA/'memory_eight_graphs.json')['entries']);assert 7 in count and 8 in count
    delivery=read(DATA/'delivery.json');assert delivery['status']=='passed'
    for doc in delivery['documents']:
        for kind in ('source','output'):
            path=ROOT.parent.parent/doc[kind]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==doc[kind+'_sha256'],path
    assert all(not r['warnings'] for r in read(DATA/'document_build.json')['results'])
    report=dict(status='passed',Werner_points=21,BB84_core_points=12,Werner_optimum_enclosures=84,BB84_optimum_enclosures=36,
        complete_Werner_instruments=6895,selected_budget_point_records=len(result['protocols']),exported_ED_definitions=len(catalog['ed_definitions']),
        certified_Werner_selected_positive=sum(r['rate']>0 for r in result['six_state']),BB84_core_policy_cells=policy_cells,
        new_large_graph_memory_distribution=dict(count),maximum_Werner_core_absolute_gap=max(r['upper']-r['lower'] for r in result['werner_complete_core']),
        maximum_BB84_core_absolute_gap=max(r['certified_upper']-r['lower'] for r in result['bb84_complete_core']),
        delivered_documents=delivery['documents'],python=sys.version,numpy=np.__version__,platform=platform.platform(),
        scope='Audit of registered finite classes, explicit constructions, certificates and reconstruction records. No unrestricted CRO converse or infinite-protocol optimality claim.')
    (DATA/'completion_audit.json').write_text(json.dumps(report,indent=2))
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(DATA.glob('*.json'))+list(DATA.glob('*.csv'))+list((ROOT/'figures').glob('*.tex'))
    paths+=[OLD/'algebra.py',OLD/'rates.py',OLD/'bernstein.py',OLD/'interval_certificate.py',OLD/'data/blocks.json',OLD/'data/codes.json',EXT/'data/codes.json']
    project=ROOT.parent.parent
    paths+=[project/'manuscript/main.tex',project/'manuscript/references.bib']
    paths+=[project/doc['output'] for doc in delivery['documents']]
    manifest={str(p.relative_to(ROOT.parent.parent)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (ROOT/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
