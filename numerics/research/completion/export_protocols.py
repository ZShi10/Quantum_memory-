"""Self-contained selected ED/AD definitions with current exact memory."""
from common import *
import json
from verify_streaming import solve_combination
from streaming_clifford import multiply


def main():
    current=json.loads((DATA/'current_results.json').read_text());old=json.loads((EXT/'data/protocol_catalog.json').read_text())
    small=json.loads((DATA/'small_memory_exact.json').read_text());graph=json.loads((DATA/'graph_memory_exact.json').read_text())
    complete=json.loads((DATA/'werner_complete_library.json').read_text());lc={e['id']:e for e in complete['entries']}
    large=json.loads((DATA/'memory_eight_graphs.json').read_text())
    definitions={e['id']:e for e in json.loads((OLD/'data/blocks.json').read_text())['entries']}
    definitions.update({e['id']:e for e in json.loads((OLD/'data/graph_family_blocks.json').read_text())['entries']})
    definitions.update({e['id']:e for e in json.loads((EXT/'data/graphs.json').read_text())['entries']})
    definitions.update(old['ed_definitions']);definitions.update({e['id']:e for e in large['entries']})
    needed=set()
    for r in current['protocols'].values():
        needed.update(r.get('EDs',[]))
        if 'ED' in r:needed.add(r['ED'])
    out={}
    for eid in sorted(needed):
        if eid in lc:
            e=lc[eid];gg=complete['graphs'][e['graph']]
            out[eid]={**e,'edges':gg['edges'],'streaming':gg['compilation']}
        elif eid in small['entries']:
            e=definitions[eid];c=small['measurements'][small['entries'][eid]['measurement_key']];rows=c['generators'];maps=[]
            for target in [e['logical_Z']]+e['checks']:
                mask=solve_combination(rows,target);product=[0,0,0]
                for j,p in enumerate(rows):
                    if mask>>j&1:product=multiply(product,[p,0,1<<j],e['m'])
                assert product[0]==target
                measured=solve_combination([r['original_generator_mask'] for r in c['measurements']],mask)
                offset=product[1]
                for j,r in enumerate(c['measurements']):
                    if measured>>j&1:offset^=r['offset']
                maps.append(dict(measurement_index_mask=measured,offset=offset))
            out[eid]=dict(id=eid,m=e['m'],M=c['M'],real=c['real'],checks=e['checks'],logical_X=e['logical_X'],logical_Z=e['logical_Z'],streaming=c,
                          key_then_checks=maps,note='Masks index measurements in event order, not physical slot numbers. Offsets give positive Hermitian original observables; relative Alice/Bob offsets cancel. Bob uses the conjugate circuit.')
        elif eid.startswith(('XG','G','LARGE')):
            e=definitions[eid];gid=eid.split('_')[0] if eid.startswith('XG') else eid
            c=large['graphs'][e['graph']]['compilation'] if eid.startswith('LARGE') else graph['graphs'][graph['entry_to_graph'][gid]]
            keep=('m','edges','logical_dual_vector','local_H_pattern','input_rotations','input_axis_permutation','input_axis_gates_on_each_wire','axis_permutation','axis_gates','ED_acceptance')
            out[eid]={k:e[k] for k in keep if k in e};out[eid].update(id=eid,M=c['M'],streaming=c,
                instruction='When each wire is loaded, apply the stated input-axis gates and per-wire H, then follow streaming events. Never preload all wires to apply local gates. Invert the recorded affine masks to recover graph-basis bits; form key/checks with the dual vector. Bob uses complex-conjugate gates.')
        elif eid.startswith('restart_pump_'):
            r=int(eid.rsplit('_',1)[1]);out[eid]=dict(family='raw-pair restart pumping',r=r,M=1 if r==1 else 2,
                procedure='Retain the control pair after each successful bilateral CNOT comparison with a fresh raw target measured in Z. A failed comparison discards the control and restarts. After r inputs on the success path, apply bilateral H and measure Z. Follow the recorded pooled B prefix and final AD rule.',
                cost='[1+sum_{j=1}^{r-1}((1-Q)^j+Q^j)]/[(1-Q)^r+Q^r] raw pairs per accepted output; this includes every failed attempt.')
        else:
            assert eid in definitions,eid
            out[eid]=definitions[eid]
    codes={e['id']:e for e in json.loads((OLD/'data/codes.json').read_text())['entries']}
    codes.update({e['id']:e for e in json.loads((EXT/'data/codes.json').read_text())['entries']})
    conventions={**old['conventions'],
        'memory':'Exact minimum for the specified complete terminal measurement; incoming port and working registers included. Pilot M=2 for real or M=1 for full. No global rate-optimal memory claim.',
        'graph_gate_order':'Local input rotations are performed individually at input load, followed by the current signed streaming schedule. Historical full-terminal schedules are superseded.',
        'canonical_Bell_frames':'LC branch bell_permutation puts raw probabilities into the stored primitive order. A bit-group swap is a classical key flip; phase relabelings conditional on the logical bit do not change joint entropy and require no additional quantum storage.',
        'policy_sources':'Core and extended policy certificate files contain exact closed t intervals, retained AD cosets and rates. The selected record links its source. For fixed pumping, no parameter-dependent choice is made.'}
    result=dict(conventions=conventions,ed_definitions=out,code_definitions=codes,protocols=current['protocols'])
    (DATA/'protocol_catalog.json').write_text(json.dumps(result,separators=(',',':')))
    print('Current protocol catalog',len(out),'ED definitions',len(codes),'code definitions',len(current['protocols']),'selected budget/point records',flush=True)


if __name__=='__main__':main()
