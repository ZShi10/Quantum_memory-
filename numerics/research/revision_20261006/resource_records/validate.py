"""Independent raw-cost identities, actual-witness replay, and signed block audit."""
from export import *
from itertools import product
import time
from graph_families import graph_instruments,gf2_rank
from recurrence_v2 import prepare as old_prepare,memory as old_memory,step as old_step
from resource_model.model import Node,step as tree_step


def rank_memory(m,edges):
    adjacency=[0]*m
    for a,b in edges:adjacency[a]|=1<<b;adjacency[b]|=1<<a
    full=(1<<m)-1;dp=[m+1]*(1<<m);dp[0]=0
    for s in range(full):
        cutrank=gf2_rank([adjacency[i]&(full^s) for i in range(m) if s>>i&1])
        for i in range(m):
            if not s>>i&1:dp[s|(1<<i)]=min(dp[s|(1<<i)],max(dp[s],cutrank+1))
    return dp[-1]


def validate_definitions(data):
    checks=dict(compiler_bitstring_reconstructions=0,gate_live_wire_checks=0,exact_flag_polynomial_checks=0,
                independent_rank_memory_checks=0,old_pivot_mismatches=0,old_nonzero_retained_flag_mismatches=0)
    definitions={r['definition']['ED']:r['definition'] for r in data['witnesses'] if r.get('definition',{}).get('ED')}
    old=json.loads((PROJECT/DEFAULT_FIXED).read_text())
    for name,d in definitions.items():
        m=d['m'];graph=d['graph'];comp=graph['compilation'];M=d['terminal_M'];live=set();peak=0
        assert rank_memory(m,graph['edges'])==M==comp['M_min'];checks['independent_rank_memory_checks']+=1
        for event in comp['events']:
            if event['action']=='load':
                assert event['wire'] not in live;live.add(event['wire']);peak=max(peak,len(live));assert sorted(live)==event['live']
            elif event['action']=='gate':
                assert set(event['gate'][1:])<=live;checks['gate_live_wire_checks']+=1
            elif event['action']=='measure_Z':assert event['wire'] in live;live.remove(event['wire'])
            else:raise AssertionError(event)
        assert not live and peak==M
        for original in range(1<<m):
            measured=sum((((original&a['original_generator_mask']).bit_count()%2)^a['offset'])<<i for i,a in enumerate(comp['measurements']))
            restored=sum((((measured&a['measured_outcome_mask']).bit_count()%2)^a['affine_offset'])<<i for i,a in enumerate(d['signed_affine_reconstruction']))
            assert restored==original
            v=d['logical_dual_vector'];key=(original>>d['key_original_generator_index'])&1
            representative=original^(key*v);pivot=d['key_original_generator_index']
            expected=(representative&((1<<pivot)-1))|((representative>>(pivot+1))<<pivot)
            actual=sum(((original&h).bit_count()%2)<<i for i,h in enumerate(d['public_check_generator_masks']))
            assert actual==expected;checks['compiler_bitstring_reconstructions']+=1
        counts=next(graph_instruments(m,graph['edges'],[d['logical_dual_vector']]))[1]
        weights=np.load(PROJECT/f'research/completion/data/werner_states_m{m}.npz')['weights']
        for branch in d['all_flags']:
            assert np.array_equal(counts[branch['syndrome']][branch['bell_permutation']],branch['factor']*weights[branch['state']])
            checks['exact_flag_polynomial_checks']+=1
        oldpivot=old['definitions'][name]['key_original_generator_index']
        if oldpivot!=pivot:
            checks['old_pivot_mismatches']+=1
            checks['old_nonzero_retained_flag_mismatches']+=sum(any(b['syndrome'] for b in row['kept_branches']) for row in old['protocols'] if row['ED']==name)
    return checks


def legacy_cost_checks():
    checks=0;maximum_error=0.
    states=[np.array([.55,.12,.18,.15]),np.array([.7,.1,.1,.1]),np.array([.6,.1,0,.3])]
    for depth in range(5):
        for letters in product('ZXRC',repeat=depth):
            word=''.join(letters);tree=homogeneous_tree(word)
            for axis in ('Z','X','Y'):
                record=dict(family='real_decoder_recurrence',path=word,final_axis=axis,memory_upper_bound=old_memory(word,axis))
                assert terminal_memory(tree,axis)==record['memory_upper_bound']
                for p in states:
                    out,y=old_prepare(p,record);actual=evaluate(tree,p,axis)
                    maximum_error=max(maximum_error,float(np.max(np.abs(out-actual['state']))),abs(float(y)-float(actual['yield_per_raw_pair'])))
                    assert np.allclose(out,actual['state'],atol=1e-13,rtol=0)
                    # Independent expansion of homogeneous renewal costs.
                    current=p.copy();denominator=1.
                    for decoder in word:current,s=old_step(current,current,decoder);denominator*=float(s)
                    cost=(2**depth)/denominator
                    assert abs(cost-float(actual['expected_raw_cost']))<2e-12*max(1,cost);checks+=1
    for primitive in 'ZXRC':
        for r in (1,2,3,5,9,17,32):
            for axis in ('Z','X','Y'):
                tree=pumping_tree(primitive*(r-1));record=dict(family='real_decoder_pumping',primitive=primitive,r=r,final_axis=axis)
                for p in states:
                    out,y=old_prepare(p,record);actual=evaluate(tree,p,axis);current=p.copy();prefix=1.;numerator=1.
                    for _ in range(r-1):
                        numerator+=prefix;current,s=old_step(current,p,primitive);prefix*=float(s)
                    cost=numerator/prefix
                    assert np.allclose(out,actual['state'],atol=1e-13,rtol=0)
                    assert abs(cost-float(actual['expected_raw_cost']))<2e-12*max(1,cost)
                    assert abs(float(y)-float(actual['yield_per_raw_pair']))<1e-13;checks+=1
    return dict(legacy_state_memory_and_expanded_cost_checks=checks,maximum_legacy_state_or_yield_error=maximum_error)


def asymmetric_renewal_checks():
    """Independently compute unnormalized yield denominators with Fractions."""
    from recurrence_v2 import MAPS
    p=(F(11,20),F(3,25),F(9,50),F(3,20))
    leaves=[RAW,homogeneous_tree('R'),pumping_tree('ZX'),homogeneous_tree('XR')]
    def algebra(node):
        if isinstance(node,Raw):return p,F(1),F(1)
        a,pa,da=algebra(node.left);b,pb,db=algebra(node.right);raw=[F(0)]*4
        for i,j,k in MAPS[node.decoder]:raw[k]+=a[i]*b[j]
        mass=sum(raw);den=da*pb+db*pa
        return raw,mass,den
    count=0
    for left in leaves:
        for right in leaves:
            for decoder in 'ZXRC':
                tree=Node(decoder,left,right);raw,mass,den=algebra(tree);out=evaluate_exact(tree,p)
                assert out['expected_raw_cost']==den/mass
                assert out['state']==tuple(x/mass for x in raw)
                cl=evaluate_exact(left,p)['expected_raw_cost'];cr=evaluate_exact(right,p)['expected_raw_cost']
                assert out['expected_raw_cost']==(cl+cr)/out['success'];count+=1
    return dict(asymmetric_exact_renewal_and_unnormalized_denominator_checks=count,
      renewal_identity='E[C_total]=E[C_attempt]+(1-s)*E[C_total]; C_attempt=C_left+C_right. Correlation between attempt cost and its success does not invalidate this renewal identity.')


def actual_witness_checks(data):
    checks=0;minimum_OTP_term=0.;max_werner_difference=0.;definitions=set()
    for row in data['witnesses']:
        if row.get('status')=='discard':continue
        d=row['definition']
        if 'resource_contract' not in d:continue
        ed=d['source_ED_record'];tree=as_tree(ed);axis=ed['final_axis'];budget=int(row['budget'].split('M')[-1])
        assert terminal_memory(tree,axis)<=budget
        assert d['terminal_schedule']['peak']==d['resource_contract']['terminal_memory']
        assert d['retained_output_schedule']['peak']==d['resource_contract']['retained_memory']
        for probe in row['illustrative_cost_replays']:
            C=F(probe['exact_expected_raw_pairs_per_ED_output']);ledger=probe['ledger']
            assert abs(float(1/C)-ledger['ED_output_bits_per_raw_pair'])<1e-14
            assert abs(ledger['gross_key_bits_per_raw_pair']-ledger['OTP_mask_bits_per_raw_pair']-ledger['net_key_bits_per_raw_pair'])<1e-13
            for term in ledger['terms']:
                minimum_OTP_term=min(minimum_OTP_term,term['net_per_raw_pair'])
            checks+=1
            if row['model']=='Werner':
                lo,hi=row['saved_rate_certificate']['lower'],row['saved_rate_certificate']['upper'];rate=ledger['net_key_bits_per_raw_pair']
                if not lo-3e-12<=rate<=hi+3e-12:raise AssertionError((row['source_key'],rate,lo,hi))
                max_werner_difference=max(max_werner_difference,abs(rate-(lo+hi)/2))
        for item in row['AD_strategy']:
            definition=item['executable_AD']
            if 'generator_rows' in definition:
                G,H,n=definition['generator_rows'],definition['parity_check_rows'],definition['n']
                assert gf2_rank(G)==definition['k'] and gf2_rank(H)==n-definition['k']
                assert all((g&h).bit_count()%2==0 for g in G for h in H)
                definitions.add(json.dumps(definition,sort_keys=True))
    return dict(actual_witness_probe_checks=checks,distinct_AD_matrix_definitions_checked=len(definitions),
      maximum_werner_saved_certificate_midpoint_difference=max_werner_difference,
      minimum_signed_OTP_or_other_component_per_raw_pair=minimum_OTP_term)


def main():
    start=time.time();path=HERE/'witnesses.json';encoded=path.read_bytes();data=json.loads(encoded)
    result=dict(status='passed',witness_sha256=hashlib.sha256(encoded).hexdigest(),
      witness_count=len(data['witnesses']),**validate_definitions(data),**legacy_cost_checks(),
      **asymmetric_renewal_checks(),**actual_witness_checks(data),seconds=time.time()-start)
    atomic_json(HERE/'validation.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
