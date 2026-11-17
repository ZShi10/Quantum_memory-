"""Cross-artifact release checks; this does not replace numerical proofs."""
from core import *
import argparse,math,hashlib


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');a=parser.parse_args()
    issues=[];checks={};files={}
    def read(path):
        path=Path(path)
        if not path.exists():issues.append('Missing '+str(path.relative_to(PROJECT)));return None
        files[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def check(ok,label):
        checks[label]=bool(ok)
        if not ok:issues.append(label)
    def intervals(rows,lo='lower',hi='upper'):
        return all(math.isfinite(float(v[lo])) and math.isfinite(float(v[hi])) and
                   0<=float(v[lo])<=float(v[hi])+1e-16 for v in rows)
    core=read(DATA/'werner_joint_union_core_certified.json')
    check(core is not None and set(core['results'])=={'0.1','0.2','0.25','0.28','0.3','0.32'},'Six complete Werner core anchors')
    for permissions in ('real','full'):
        rec=read(DATA/f'werner_v2_{permissions}_menu_certified.json')
        check(rec is not None and len(rec['results'])==6 and all(intervals(v['results'].values()) for v in rec['results'].values()),f'Werner {permissions} recurrence intervals')
    for permissions,qs in [('real',(.15,.18,.2,.22,.24)),('full',(.15,.2,.22,.24))]:
        for q in qs:
            cert=read(DATA/f'bb84_joint_extension_{permissions}_{q:.5f}_{q:.5f}_n128_menu_certified.json')
            check(cert is not None and cert.get('complete') and intervals(cert['results'].values()),f'BB84 {permissions} Q={q} complete menu bounds')
            if permissions=='real':
                block=read(DATA/f'bb84_block_relaxation_real_{q:.5f}_{q:.5f}_n128.json')
                check(block is not None and block.get('complete'),f'BB84 real Q={q} complete block relaxation')
    curve=read(DATA/'bb84_policy_curves_real.json');catalog=read(DATA/'bb84_curve_policies_real.json')
    expected={f'{i/2000:.8f}_{i/2000:.8f}' for i in range(501)}
    check(curve is not None and set(curve['results'])==expected,'BB84 501-point grid')
    if curve and catalog:
        check(all(v.get('catalog_digest')==catalog['digest'] for v in curve['results'].values()),'Every BB84 point uses final catalog')
        check(all(all(math.isfinite(float(r['lower'])) and 0<=r['lower']<=1 for r in v['results'].values()) for v in curve['results'].values()),'Finite achievable curve rates')
        check(all(all(v['results'][f'M{i}']['lower']<=v['results'][f'M{i+1}']['lower'] for i in range(1,8)) for v in curve['results'].values()),'BB84 memory-budget nesting')
    for model,N in [('werner',500),('bb84',501)]:
        control=read(DATA/f'du_Fig15_{model}_recomputed.json')
        check(control is not None and control.get('complete') and len(control['results'])==N,f'Du {model} reconstruction complete')
        if control and model=='bb84':
            check(all(control['results'].get(f'{i/2000:.12g}',{}).get('Q')==i/2000 for i in range(501)), 'Exact common BB84 plotting grid')
    summary=read(DATA/'joint_union_summary.json')
    check(summary is not None and len(summary['results'])==88 and intervals(summary['results']),'88 union intervals in tables')
    protocols=read(DATA/'fixed_block_protocols.json')
    if protocols:
        check(all(v['certificate'] is not None for v in protocols['protocols']),'Every exported fixed-block witness has its own matching certificate')
        for name,definition in protocols['definitions'].items():
            c=definition['graph']['compilation'];check(c['M']==definition['M'],name+' register schedule')
    separation=read(DATA/'single_round_all_modes_certified.json')
    check(separation is not None and separation['chosen_rate_lower']>separation['endpoint_all_lengths_all_modes_and_W_upper'],'Strict intermediate-dimension code separation')
    for name in ('terminal_B_equivalence_checks.json','coarse_spc_checks.json','zero_coordinate_checks.json'):
        v=read(DATA/name);check(v is not None and v.get('status')=='passed',name)
    for name in ('representative_protocols.json','attained_target_rates.json','controlled_examples_certified.json','bb84_asymmetric_grid.json','result_manifest.json'):
        read(DATA/name)
    for name in ('werner_comparison','bb84_comparison','bb84_regions'):
        provenance=read(ROOT/f'figures/{name}_provenance.json')
        check(provenance is not None and (ROOT/f'figures/build/{name}.pdf').exists(),name+' figure and provenance')
    for directory in (ROOT/'draft',ROOT):
        for p in directory.glob('*.tex' if directory.name=='draft' else '*.py'):
            files[str(p.relative_to(PROJECT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=dict(status='passed' if not issues else 'incomplete',checks=checks,issues=issues,file_sha256=files,
             note='Cross-file completeness, association, finite arithmetic and provenance audit. Scientific certificates and visual PDF review are separate requirements.')
    atomic_json(DATA/'final_artifact_audit.json',out)
    print('Final artifact audit:',out['status'],len(checks),'checks;',len(issues),'issues',flush=True)
    for issue in issues:print(' -',issue,flush=True)
    if a.require_complete:assert not issues,'Final release prerequisites incomplete'


if __name__=='__main__':main()
