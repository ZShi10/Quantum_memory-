"""Read actual saved witnesses; never reselect a protocol or infer an upper bound."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,argparse
from fractions import Fraction as F
from functools import lru_cache
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
for relative in ('research/joint_optimization','research/completion','research/revision_20260922','research/revision_20261006'):
    sys.path.insert(0,str(PROJECT/relative))
import numpy as np
from resource_model.model import (tree_from_dict,homogeneous_tree,pumping_tree,structural_record,
    executable_schedule,evaluate,evaluate_exact,Evaluator,RAW,Raw,canonical,ancestry,terminal_memory)
from core import b_step
from ad_modes import OTPKernel,repetition_components,spc_otp_components
import compound_terminals as ct
from code_extension import api as codes_api

DEFAULT_WERNER='research/revision_20261006/werner_search/runs/058a29e3a147c4101840/data/werner_v2_full_certified.json'
DEFAULT_BB84='research/revision_20261006/bb84_search/runs/705a628c40d89588c33f0d3f613d11b5b3b2a383b89c302e75c57eb958ff6371/data/bb84_joint_extension_real_0.23000_0.23000_n128.json'
DEFAULT_FIXED='research/revision_20260922/data/fixed_block_protocols.json'


def snapshot(path):
    path=Path(path);path=path if path.is_absolute() else PROJECT/path
    encoded=path.read_bytes()
    return json.loads(encoded),dict(path=path.relative_to(PROJECT).as_posix(),sha256=hashlib.sha256(encoded).hexdigest())


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.writing.json')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)


def as_tree(record):
    family=record['family']
    if family=='explicit_binary_tree':return tree_from_dict(record['tree'])
    if family=='real_decoder_recurrence':return homogeneous_tree(record['path'])
    if family=='real_decoder_pumping':return pumping_tree(record['primitive']*(record['r']-1))
    raise ValueError(f'Unregistered ED family: {family}')


@lru_cache(None)
def selected_kernel(code):
    """Read an existing kernel, or construct in memory; never write old/new caches."""
    entry=next(e for e in codes_api.entries(all_codes=True) if e['id']==code)
    for root in (codes_api.DATA/'kernels',PROJECT/'research/resource_extension/data/kernels'):
        path=root/f'{code}.npz'
        if path.exists():return codes_api._load_kernel(path,entry)
    return codes_api.build_kernel(entry)


def kernel_dictionary():
    # compound_terminals.values requests only its selected action's code.
    class Lazy(dict):
        def __missing__(self,key):self[key]=selected_kernel(key);return self[key]
    return Lazy()


_KERNELS=kernel_dictionary()
ct.kernels=lambda:_KERNELS


def nullspace(rows,n):
    """Deterministic GF(2) nullspace with lowest-coordinate pivots."""
    rows=list(map(int,rows));at=0;pivots=[]
    for col in range(n):
        pivot=next((i for i in range(at,len(rows)) if rows[i]>>col&1),None)
        if pivot is None:continue
        rows[at],rows[pivot]=rows[pivot],rows[at]
        for i in range(len(rows)):
            if i!=at and rows[i]>>col&1:rows[i]^=rows[at]
        pivots.append(col);at+=1
    output=[]
    for col in range(n):
        if col in pivots:continue
        value=1<<col
        for row,pivot in zip(rows,pivots):
            if row>>col&1:value|=1<<pivot
        output.append(value)
    return output


def action_definition(action):
    a={k:v for k,v in action.items() if k!='classical_B_rounds'}
    if a['kind']=='Watanabe_2007':
        return dict(**a,protocol='Two-bit parity comparison with hashed reconciliation and privacy amplification',
          reference='Watanabe et al., arXiv:0705.2904v2, Theorem 3 Eq.(4)',
          executable_specification='For local pairs A1,A2 and B1,B2, form U1=A1 xor A2 and V1=B1 xor B2. Use the cited hashed reconciliation to determine W1=U1 xor V1. Define U2=A2 if W1=0 and U2=0 otherwise, and the analogous Bob variable. Reconcile and privacy amplify the resulting variables according to the cited two-stage protocol, accounting for every public hash. No OTP mask is used.',
          rate_implementation='research/revision_20260922/literature_rates.py:watanabe',
          independent_cq_check='research/revision_20260922/data/watanabe_reference_checks.json')
    n=a['n']
    if a['kind']=='linear':entry=selected_kernel(a['code']).entry;k=entry['k'];G=entry['rows']
    elif a['kind']=='repetition':k=1;G=[(1<<n)-1]
    elif a['kind']=='SPC':k=n-1;G=[1|(1<<i) for i in range(1,n)]
    else:raise ValueError(a)
    H=nullspace(G,n)
    assert len(H)==n-k and all((g&h).bit_count()%2==0 for g in G for h in H)
    return dict(**a,k=k,generator_rows=G,parity_check_rows=H,
      coordinate_encoding='Integer bit i represents coordinate i; coordinate 0 is least significant.',
      syndrome='S=H*(Alice_bits xor Bob_bits); all observed syndromes are kept as labeled streams before reconciliation.',
      classical_protocol='Apply the registered Du code protocol; reconcile each syndrome stream, then use a joint two-universal privacy-amplification hash with the proved total entropy bound. No hidden-state-dependent stream acceptance is required.',
      reconciliation_and_syndrome_cost='All public reconciliation is included in the rate expression; OTP masks, when selected, are separately paid and subtracted.',
      OTP={'no_otp':'Public syndrome exchange; no mask consumption.',
           'otp_plain':f'Encrypt the {n-k} independent check bits per block with pre-shared uniform mask bits.',
           'otp_compressed':'Encrypt the asymptotically compressed syndrome message with pre-shared uniform masks; net cost is H(S)/n per AD input bit. The source rate does not assume a free mask supply.'}[a['mode']],
      reference='Du et al., arXiv:2404.14733v2; the registered rate model specifies each OTP and hashing variant.')


def normalized_action(action):
    out=dict(action)
    if out['kind']=='Watanabe_2007':
        out.setdefault('n',2);out.setdefault('mode','no_otp_hashed')
    return out


def terminal_components(state,action):
    state=np.asarray(state,float).reshape(1,4)
    net=float(ct.values(state,[action])[0,0]);kind=action['kind'];mode=action['mode'];n=action['n']
    if mode in ('no_otp','no_otp_hashed'):cost=0.
    elif mode=='otp_plain':
        k=selected_kernel(action['code']).k if kind=='linear' else (1 if kind=='repetition' else n-1)
        cost=(n-k)/n
    elif kind=='linear':cost=float(OTPKernel(selected_kernel(action['code']).entry).components(state)[1][0])
    elif kind=='repetition':cost=float(repetition_components(state,n)[2][0])
    else:cost=float(spc_otp_components(state,n)[1][0])
    return dict(net_per_AD_input_bit=net,gross_per_AD_input_bit=net+cost,OTP_cost_per_AD_input_bit=cost)


def cost_ledger(state,yield_ed,strategy,rounds):
    state=np.asarray(state,float);yb=1.;steps=[]
    for r in range(rounds):
        state,success=b_step(state);factor=float(success)/2;yb*=factor
        steps.append(dict(round=r+1,acceptance_probability=float(success),yield_factor=factor,
                          AD_input_bits_per_ED_output_so_far=yb))
    y=float(yield_ed)*yb;terms=[]
    for item in strategy:
        alpha=F(item['allocation_exact']);action=normalized_action(item['AD']);component=terminal_components(state,action)
        terms.append(dict(AD=action,allocation_exact=str(alpha),allocation_unit='fraction_of_bits_entering_AD',
            AD_blocks_per_raw_pair=float(alpha)*y/action['n'],
            **component,**{name.replace('_per_AD_input_bit','_per_raw_pair'):float(alpha)*y*value
                         for name,value in component.items()}))
    assert sum(F(item['allocation_exact']) for item in strategy)<=1
    return dict(ED_output_bits_per_raw_pair=float(yield_ed),classical_B_rounds=rounds,classical_B_steps=steps,
                classical_B_yield_per_ED_output=yb,AD_input_bits_per_raw_pair=y,AD_input_Bell_state=state.tolist(),
                terms=terms,gross_key_bits_per_raw_pair=sum(t['gross_per_raw_pair'] for t in terms),
                OTP_mask_bits_per_raw_pair=sum(t['OTP_cost_per_raw_pair'] for t in terms),
                net_key_bits_per_raw_pair=sum(t['net_per_raw_pair'] for t in terms),
                paid_OTP_net_terms_are_not_clipped=True)


def node_cost_records(tree,p):
    evaluator=Evaluator(p);out={}
    def visit(node):
        name=canonical(node)
        if name in out:return
        data=evaluator.retained(node)
        if not isinstance(node,Raw):visit(node.left);visit(node.right)
        out[name]=dict(decoder=None if isinstance(node,Raw) else node.decoder,
          successful_ancestry_inputs=ancestry(node),state=data['state'].tolist(),
          parent_attempt_acceptance_probability=float(data['success']),
          mean_consumed_raw_pairs=float(data['expected_raw_cost']),
          cost_equation='1' if isinstance(node,Raw) else '(C_left+C_right)/parent_success',
          children=[] if isinstance(node,Raw) else [canonical(node.left),canonical(node.right)])
    visit(tree);return out


def tree_definition(ed):
    tree=as_tree(ed);axis=ed['final_axis'];permission='enlarged_axes' if axis=='Y' or ed.get('permissions')=='enlarged_axes' else 'real'
    record=structural_record(tree,axis,permission)
    assert record['memory_upper_bound']==ed['memory_upper_bound']
    return tree,dict(source_ED_record=ed,resource_contract=record,terminal_schedule=executable_schedule(tree,axis),
                    retained_output_schedule=executable_schedule(tree,None),
                    cost_identity='C_T=(C_left+C_right)/s_T; C_raw=1',
                    physical_retry_explanation='Prepare one successful output from each child, retrying a failed child without discarding the other successful child. A rejected parent discards both child outputs and restarts. The mean raw cost is the renewal-reward ratio of mean cost per parent attempt to parent success probability; cost and success within an attempt need not be independent.',
                    terminal_and_retained_output_are_different_contracts=True)


def exact_source(model,Q=None,b=None,p=None,t=None):
    if model=='Werner':
        q=F(str(Q));return (1-F(3,2)*q,q/2,q/2,q/2)
    b,p,t=F(str(b)),F(str(p)),F(str(t));lo=max(F(0),b+p-1);hi=min(b,p);z=lo+(hi-lo)*t
    return (1-b-p+z,b-z,z,p-z)


def tree_witness(row,source,model,b=None,p=None,key=None):
    if 'ED' not in row:return dict(model=model,source=source,source_key=key,status='discard',lower=row.get('lower',0))
    tree,definition=tree_definition(row['ED']);axis=row['ED']['final_axis']
    if model=='Werner':
        rounds=row['AD'].get('classical_B_rounds',0);strategy=[dict(AD=row['AD'],allocation_exact='1')];probes=[None]
    else:
        rounds=row.get('classical_B_rounds',row.get('AD',{}).get('classical_B_rounds',0))
        strategy=row.get('strategy') or [dict(AD=row['AD'],allocation_exact='1')]
        argmin=F(str(row.get('certificate',{}).get('argmin',0.)))
        assert 0<=argmin<=1, 'Certificate probe lies outside the BB84 correlation domain.'
        probes=sorted(set([0.,.25,.5,.75,1.,float(argmin)]))
    strategy=[{**item,'AD':normalized_action(item['AD'])} for item in strategy]
    numerical=[]
    for t in probes:
        source_state=exact_source(model,Q=row.get('Q'),b=b,p=p,t=t)
        exact=evaluate_exact(tree,source_state,axis);floating=evaluate(tree,np.array(list(map(float,source_state))),axis)
        assert np.max(np.abs(floating['state']-np.array(list(map(float,exact['state'])))))<1e-12
        assert abs(float(exact['expected_raw_cost'])-float(floating['expected_raw_cost']))<=1e-10*max(1,float(exact['expected_raw_cost']))
        cost=float(exact['expected_raw_cost']);yield_ed=float(exact['yield_per_raw_pair'])
        if model=='Werner' and 'expected_raw_pairs_per_ED_output' in row:
            assert abs(cost-row['expected_raw_pairs_per_ED_output'])<1e-10*max(1,cost)
        ledger=cost_ledger(exact['state'],yield_ed,strategy,rounds)
        numerical.append(dict(t=t,raw_Bell_state=list(map(float,source_state)),ED_Bell_state=list(map(float,exact['state'])),
          expected_raw_pairs_per_ED_output=cost,exact_expected_raw_pairs_per_ED_output=str(exact['expected_raw_cost']),
          unique_node_costs=node_cost_records(tree,np.array(list(map(float,source_state)))),ledger=ledger))
    return dict(model=model,source=source,source_key=key,budget=row.get('budget',key),
      source_parameters={'Q':row['Q']} if model=='Werner' else dict(bit_error=b,phase_error=p,
        hidden_z='max(0,b+p-1)+(min(b,p)-max(0,b+p-1))*t; 0<=t<=1'),
      definition=definition,AD_strategy=[dict(**item,executable_AD=action_definition(item['AD'])) for item in strategy],
      classical_B_rounds=rounds,classical_B_protocol='Pair disjoint classical outcomes, publish only local pair parities, retain one outcome when relative parity is zero. Each round multiplies yield by its success probability divided by two.',
      saved_rate_certificate=row.get('certificate',dict(lower=row.get('lower'),upper=row.get('upper'),scope=row.get('scope'))),
      certified_lower_from_source=row.get('lower'),illustrative_cost_replays=numerical,
      probe_scope='Exact-rational ED cost replay and floating classical rate diagnostics. BB84 probes are not extrema or a replacement for the saved whole-interval certificate.',
      no_new_global_upper_bound_claim=True)


def affine_reconstruction(compilation,m):
    measurements=compilation['measurements'];rows=[int(a['original_generator_mask']) for a in measurements]
    lookup={0:0}
    for i,row in enumerate(rows):lookup.update({value^row:mask^(1<<i) for value,mask in list(lookup.items())})
    assert len(lookup)==1<<m
    offsets=sum(int(a['offset'])<<i for i,a in enumerate(measurements))
    return [dict(original_generator_index=i,measured_outcome_mask=lookup[1<<i],
                 affine_offset=(lookup[1<<i]&offsets).bit_count()%2) for i in range(m)]


def frame_bits(permutation):
    labels=((0,0),(1,0),(1,1),(0,1));mapping=[labels[permutation.index(i)] for i in range(4)]
    for c in (0,1):
        for a in (0,1):
            for d in (0,1):
                if all(out==(x^c,z^(a*x)^d) for (x,z),out in zip(labels,mapping)):
                    return dict(bit_translation=c,phase_shear=a,phase_translation=d,
                      physical_key_correction='Bob flips his private key bit iff bit_translation=1; phase entries are virtual rate frames.')
    raise AssertionError(permutation)


def fixed_definition(name,entry,graph):
    m=entry['m'];v=entry['logical_dual_vector'];pivot=v.bit_length()-1
    checks=[(1<<i)^(((v>>i)&1)<<pivot) for i in range(m) if i!=pivot]
    return dict(ED=name,m=m,terminal_M=entry['M'],memory_proof_kind='saved_exact_minimum_for_this_fixed_complete_terminal_measurement_over_streaming_input_orders',
      quantum_output_memory_minimum_not_claimed=True,
      raw_cost_contract='Every fixed attempt consumes all m raw pairs exactly once. Flag s contributes P_s/m outputs per consumed raw pair; sum contributions over retained flags.',
      fixed_attempt_raw_cost=m,retry_rule='No sequential subtree retry in this fixed-block contract.',
      graph=graph,logical_dual_vector=v,key_original_generator_index=pivot,public_check_generator_masks=checks,
      pivot_convention='HIGHEST set bit of logical_dual_vector; exactly matches graph_families.graph_instruments probability tensors.',
      public_bit_order='Increasing original generator index excluding the key pivot; least-significant syndrome bit first.',
      signed_affine_reconstruction=affine_reconstruction(graph['compilation'],m),
      transcript_rule='Keep all intermediate compiler measurement outcomes private; reconstruct original generator outcomes with the signed affine map. Announce only the specified public check parities. The final key pivot remains private.',
      final_measurement='Run saved load/gate/measure events locally and the conjugate circuit for Bob. This is a complete terminal instrument, not a retained quantum output.',
      all_flags=entry['branches'])


def fixed_witness(row,source,entry,graph):
    definition=fixed_definition(row['ED'],entry,graph);m=entry['m'];q=F(str(row['Q']))
    weights=np.load(PROJECT/f'research/completion/data/werner_states_m{m}.npz')['weights'];monomials=[(1-F(3,2)*q)**(m-w)*(q/2)**w for w in range(m+1)]
    branches=[]
    for branch in row['kept_branches']:
        raw=[sum(F(int(c))*monomials[w] for w,c in enumerate(poly))*branch['factor'] for poly in weights[branch['state']]]
        mass=sum(raw);state=[x/mass for x in raw];yield_ed=mass/m
        action=branch['AD'];ledger=cost_ledger(state,float(yield_ed),[dict(AD=action,allocation_exact='1')],action.get('classical_B_rounds',0))
        branches.append(dict(**branch,flag_probability=float(mass),exact_flag_probability=str(mass),
           ED_output_bits_per_raw_pair=float(yield_ed),canonical_Bell_state=list(map(float,state)),
           canonical_frame=frame_bits(branch['bell_permutation']),executable_AD=action_definition(action),ledger=ledger))
    total=sum(branch['ledger']['net_key_bits_per_raw_pair'] for branch in branches)
    assert abs(total-row['rate_numerical'])<3e-10
    return dict(model='Werner',source=source,Q=row['Q'],budget=row['budget'],definition=definition,
       retained_flags=branches,discarded_flags='All remaining public flags.',
       raw_probability_polynomial='For canonical state j: raw_s,j = factor_s * sum_w weights[state_s,j,w]*(1-3Q/2)^(m-w)*(Q/2)^w. P_s=sum_j raw_s,j; rate=sum_s (P_s/m)*classical_rate_s.',
       total_key_bits_per_raw_pair=total,saved_rate_certificate=row['certificate'],
       no_new_global_upper_bound_claim=True)


def export(werner_paths=(),bb84_paths=(),fixed_paths=()):
    rows=[];sources=[]
    for path in werner_paths:
        doc,source=snapshot(path);sources.append(source)
        for key,row in doc['results'].items():rows.append(tree_witness(row,source,'Werner',key=key))
    for path in bb84_paths:
        doc,source=snapshot(path);sources.append(source)
        for key,row in doc['results'].items():rows.append(tree_witness(row,source,'BB84',b=doc['b'],p=doc['p'],key=key))
    if fixed_paths:
        library,library_source=snapshot('research/completion/data/werner_complete_library.json');sources.append(library_source)
        entries={e['id']:e for e in library['entries']}
        for path in fixed_paths:
            doc,source=snapshot(path);sources.append(source)
            for row in doc['protocols']:
                e=entries[row['ED']];rows.append(fixed_witness(row,source,e,library['graphs'][e['graph']]))
    return dict(status='exported_and_replayed',scope='Executable lower-witness records only; saved certificates retain their original scopes.',
      common_memory_convention='Peak simultaneously live local qubits per party, including the incoming port; no external quantum buffer. Unlimited classical storage is not quantum memory.',
      common_rate_unit='Net secret bits per consumed raw entangled pair, including rejected preparation attempts and paid OTP masks; sifting/testing asymptotic overhead excluded as in the source rate model.',
      sources=sources,witnesses=rows)


def markdown(data):
    lines=['# Executable resource witnesses','',data['scope'],'',data['common_memory_convention'],'',data['common_rate_unit'],'',
      '| Input | Parameters / budget | ED construction | Terminal M | Raw-pair accounting | Rate / status |',
      '|---|---|---|---:|---|---|']
    for row in data['witnesses']:
        if row.get('status')=='discard':continue
        d=row['definition']
        if 'resource_contract' in d:
            r=d['resource_contract'];p=row['source_parameters'];rate=row['certified_lower_from_source']
            costs=[a['expected_raw_pairs_per_ED_output'] for a in row['illustrative_cost_replays']]
            cost=f'C={costs[0]:.8g}' if row['model']=='Werner' else 'C(t) replayed at listed probes; not an extremum bound'
            lines.append(f"| {row['model']} | {p} / {row['budget']} | `{r['canonical_tree']}`, axis {r['final_axis']} | {r['terminal_memory']} | {cost}; B={row['classical_B_rounds']} | saved lower {rate:.10g} |")
        else:
            lines.append(f"| Werner | Q={row['Q']} / {row['budget']} | `{d['ED']}` | {d['terminal_M']} | fixed m={d['m']}; each flag P/m | replay {row['total_key_bits_per_raw_pair']:.10g} |")
    lines+=['','Each JSON record includes the serialized circuit/tree, retained and terminal schedules, signed measurement reconstruction where applicable, the AD generator/check matrices or primary-source protocol, exact allocations, individual B yields, gross privacy yield and OTP deductions.','',
      'Recursive M is an executable schedule bound (with exactness only in its declared schedule class); fixed-block M is the saved exact streaming minimum for that complete terminal instrument. Neither asserts the global optimum over arbitrary protocols at fixed memory.','',
      'For complete graph blocks the highest-bit key pivot is essential: it matches the probability tensors. An earlier exporter used the lowest-bit pivot; this exporter reconstructs the correct flags from the source library rather than reusing that convention.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--werner',action='append');parser.add_argument('--bb84',action='append');parser.add_argument('--fixed',action='append');parser.add_argument('--output',default=str(HERE/'witnesses.json'));args=parser.parse_args()
    data=export(args.werner or [DEFAULT_WERNER],args.bb84 or [DEFAULT_BB84,'research/revision_20260922/data/bb84_joint_extension_real_0.24000_0.24000_n128.json'],args.fixed or [DEFAULT_FIXED])
    atomic_json(args.output,data);Path(args.output).with_suffix('.md').write_text(markdown(data),encoding='utf-8')
    print(json.dumps(dict(status=data['status'],witnesses=len(data['witnesses']),output=args.output)),flush=True)
