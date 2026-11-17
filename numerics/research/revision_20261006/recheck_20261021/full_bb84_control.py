"""Full-Clifford BB84 controls, with unchanged BB84 source information.

The baseline is an achievable fixed policy (not a claimed optimum).  The
block upper covers every complete single-output Clifford instrument through
five inputs and the same full AD menu.  Neither operation uses Y statistics
as additional parameter-estimation information.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, sys, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REV = HERE.parent
PROJECT = REV.parents[1]
LEGACY = PROJECT/'research/revision_20260922'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def bootstrap(job):
    sys.path[:0] = [str(REV/'bb84_search'), str(LEGACY)]
    import core
    core.ROOT = job
    core.DATA = job/'data'
    core.DATA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LEGACY/'fixed_policy_lp.py', job/'fixed_policy_lp.py')
    from n8_adapter import install
    install(core)
    return core


def baseline(q, optimize=True):
    job = HERE/'full_bb84/baseline'/str(q)
    file = job/'result.json'
    if file.exists() and json.loads(file.read_bytes()).get('complete'):
        return json.loads(file.read_bytes())
    core = bootstrap(job)
    import numpy as np
    from fractions import Fraction as F
    import bb84_joint_extension as engine
    from bb84_recursive import prepare
    from literature_rates import watanabe
    ts = np.linspace(0, 1, 257)
    candidates = []
    action = dict(kind='Watanabe_2007', n=2, mode='no_otp_hashed')
    strategy = [dict(AD=action, allocation=1., allocation_exact='1')]
    start = time.monotonic()
    # Each basis and each B prefix is fixed before the unknown t=z/Q.
    for axis in ('Z', 'X', 'Y'):
        ed = dict(family='real_decoder_recurrence', path='', final_axis=axis,
                  depth=0, memory_upper_bound=1, minimal_memory_proved=True,
                  successful_ancestry_inputs=1)
        for B in range(17):
            state, y = prepare(q, q, ts, ed, B)
            lower_sample = float((y*watanabe(state)).min())
            candidates.append((lower_sample, axis, B, ed))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = dict(lower=0., strategy=[], ED=None)
    # One candidate per basis is expanded to the complete menu. This is a
    # feasible baseline only: no global optimality is inferred from ranking.
    for axis in ('Z', 'Y'):
        _, _, B, ed = next(row for row in candidates if row[1] == axis)
        cert = engine.certify_policy(q, q, ed, B, strategy, atol=1e-10)
        row = dict(lower=max(0., cert['lower']), ED=ed, B=B,
                   strategy=strategy, certificate=cert)
        if optimize:
            candidate = engine.solve(q, q, ed, B, nmax=128, tolerance=2e-9,
                                     max_iterations=24, incumbent=0.)
            candidate.update(ED=ed, B=B)
            if candidate['lower'] > row['lower']:
                row = candidate
        if row['lower'] > best['lower']:
            best = row
        save(job/f'basis_{axis}.json', row)
    result = dict(complete=True, Q=q, b=q, p=q,
        results={f'M{m}': best for m in range(1,9)},
        best=best, seconds=time.monotonic()-start, expanded_AD=optimize,
        scope='Achievable single-input Pauli-measurement baseline; fixed B selected by a Watanabe discovery scan. If expanded_AD is true, the full n<=8/long endpoint AD menu is additionally searched at that fixed B. Not a global baseline optimum. Continuous full hidden-correlation lower certificate.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(file, result)
    print('FULL BB84 BASELINE', q, best['lower'], best.get('ED'), best.get('B'), flush=True)
    return result


def block_upper(q, compact=False):
    task = HERE/('full_bb84/block_upper_compact' if compact else 'full_bb84/block_upper')/str(q)
    receipt = task/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_bytes()).get('complete'):
        return json.loads(receipt.read_bytes())
    source = REV/'code_extension/bb84_m5/audit_upper.py'
    code = source.read_text(encoding='utf-8')
    old = "job=HERE/'upper_runs'/tag;job.mkdir(parents=True,exist_ok=True)"
    assert code.count(old) == 1
    ns = dict(__name__='full_control_upper', __file__=str(source),
              RECHECK_OUTPUT=task)
    exec(compile(code.replace(old,"job=RECHECK_OUTPUT/tag;job.mkdir(parents=True,exist_ok=True)"),str(source),'exec'), ns)
    # Full, not real: phases and all effective Pauli readouts are included.
    lower_file = HERE/'full_bb84/baseline'/str(q)/'result.json'
    args = argparse.Namespace(p=None, permissions='full', nmax=128, K=16,
        tolerance=1e-9, quantum=1e-12, lower_file=[str(lower_file)] if lower_file.exists() else [],
        test_t=['0', '.0625', '.125', '.1875', '.25', '.375', '.5', '.75', '1'])
    if compact:
        original=ns['lower_sources']
        def lower_sources(*aa,**kw):
            lower,ts,provenance=original(*aa,**kw)
            # Selection of legal test states need not itself be certified.
            # These are evaluated outward, with the SAME state for all flags.
            t='0.25'
            if lower_file.exists():
                candidate=json.loads(lower_file.read_bytes())['best'].get('certificate',{}).get('argmin')
                if candidate is not None and 0<=candidate<=1:t=str(candidate)
            return lower,sorted(set(['0',t]),key=float),provenance
        ns['lower_sources']=lower_sources
        args.test_t=[]
    result = ns['run'](args,q)
    out = dict(complete=result['complete'], Q=q,
               results=result['results'], instruments=len(result['entries']),
               output=str(task/result['job_fingerprint']/'result.json'),
               scope='Complete full-Clifford N<=5, all public flags, full-input cost, all n<=8 codes, repetition/SPC<=128, all three OTP modes, Watanabe, B<=16. Min over legal test states is an upper relaxation of a single fixed policy, not an attainable curve.')
    save(receipt,out)
    return out


def registered_upper(q):
    """Recompute all XYZ-terminal registered bounds from fresh lower seeds.

    The complete small-block Clifford component is separate. Larger circuits
    still use the explicitly registered four real decoders; allowing an XYZ
    terminal is not mislabeled as arbitrary full-Clifford recursive ED.
    """
    task=HERE/'full_bb84/registered_upper'/str(q)
    task.mkdir(parents=True,exist_ok=True)
    receipt=task/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_bytes()).get('complete'):
        return json.loads(receipt.read_bytes())
    sys.path.insert(0,str(REV/'bb84_search'))
    import run_point
    args=run_point.parser().parse_args(['--q',str(q),'--permissions','full','--short-max','8','--ed-extension'])
    save(task/'manifest.json',dict(specification=run_point.exact_spec(args)))
    shutil.copyfile(LEGACY/'fixed_policy_lp.py',task/'fixed_policy_lp.py')
    core,engine,upper,active=run_point.bootstrap(task)
    points=json.loads((PROJECT/'revised/reproduction/data/inherited/frozen_curve_delivery.json').read_bytes())['points']
    nearest=min((p for p in points if p['model']=='BB84_real'),key=lambda p:abs(float(p['Q'])-q))
    origin=REV/'bb84_search/runs'/nearest['registered_point_fingerprint']
    summary=json.loads((origin/'summary.json').read_bytes())
    historical=json.loads((origin/summary['source']).read_bytes())
    basefile=HERE/'full_bb84/baseline'/str(q)/'result.json'
    raw=json.loads(basefile.read_bytes())['best'] if basefile.exists() else dict(lower=0.)
    strategy=[dict(AD=dict(kind='Watanabe_2007',n=2,mode='no_otp_hashed'),allocation=1.,allocation_exact='1')]
    results={};certified={};start=time.monotonic()
    for m in range(1,9):
        proposal=historical['results'][f'M{m}']
        best=raw.copy()
        if m>1 and results[f'M{m-1}']['lower']>best['lower']:best=results[f'M{m-1}'].copy()
        ed=proposal.get('ED')
        if ed:
            B=proposal.get('classical_B_rounds',0)
            key=json.dumps([ed,B],sort_keys=True)
            if key not in certified:
                cert=engine.certify_policy(q,q,ed,B,strategy,atol=2e-9,max_cells=12000)
                certified[key]=dict(lower=max(0.,cert['lower']),ED=ed,classical_B_rounds=B,
                                    strategy=strategy,certificate=cert,origin='new_full_z_lower_at_this_Q')
            if certified[key]['lower']>best['lower']:best=certified[key]
        results[f'M{m}']=best
    # Historical numbers are not upper endpoints here. Only proposed finite
    # probability measures are reused, and every selected dual is re-evaluated.
    evaluated=historical.get('evaluated',{}).copy()
    if raw.get('sampled_dual'):evaluated['new_raw_Y']=raw
    source=task/'data'/f'full_terminal_axes_{q}.json'
    src=dict(b=q,p=q,permissions='full',nmax=128,K=16,results=results,evaluated=evaluated,
             scope=run_point.exact_spec(args),historical_upper_values_used=False,
             discovery_proposals_Q=nearest['Q'])
    save(source,src)
    upper.certify(source,tolerance=2e-9)
    dest=source.with_name(source.stem+'_menu_certified.json')
    result=json.loads(dest.read_bytes())
    assert result['complete']
    record=dict(complete=True,Q=q,results=result['results'],output=str(dest),
        lower_source=str(source),historical_upper_values_used=False,
        seconds=time.monotonic()-start,
        scope='Entire registered four-real-decoder recurrence/tree/pump family with XYZ terminals and complete AD menu. Complete full-Clifford N<=5 blocks are separately bounded. No claim of complete arbitrary-Clifford large recursive circuits.')
    save(receipt,record)
    print('XYZ REGISTERED UPPER',q,record['results'],flush=True)
    return record


def registered_refine(q, exclusion=False):
    """Sharpen every residual object with additional legal source states."""
    import math
    from fractions import Fraction as F
    import numpy as np
    root=HERE/'full_bb84/registered_upper'/str(q)
    base=json.loads((root/'receipt.json').read_bytes())
    report=json.loads(Path(base['output']).read_bytes())
    src=json.loads(Path(base['lower_source']).read_bytes())
    task=HERE/'full_bb84'/('registered_exclusion' if exclusion else 'registered_refined')/str(q)
    task.mkdir(parents=True,exist_ok=True)
    outpath=task/'receipt.json'
    if outpath.exists() and json.loads(outpath.read_bytes()).get('complete'):
        return json.loads(outpath.read_bytes())
    sys.path.insert(0,str(REV/'bb84_search'))
    import run_point
    shutil.copyfile(root/'manifest.json',task/'manifest.json')
    core,engine,upper,active=run_point.bootstrap(task)
    from certify_bb84_recursive import prepare
    from long_code_bounds import best_bounds
    from bb84_fixed import physical_kernel
    import stable_intervals as iv
    ts=[F(x) for x in ['0','.0625','.125','.1875','.2','.21875','.234375','.25',
                       '.265625','.28125','.3125','.375','.5','.75','1']]
    kernels=core.load_codes();physical={k.entry['id']:physical_kernel(k) for k in kernels}
    lows={key:float(v['lower']) for key,v in src['results'].items()}
    cap=5 if exclusion else 8
    screening=(lows['M5']+lows['M6'])/2 if exclusion else None
    if exclusion:assert lows['M6']>lows['M5']
    rounds=[];start=time.monotonic()
    for r in report['rounds']:
        residual=[]
        for row in r['unresolved']:
            ed=row['ED'];B=r['B'];memory=ed['memory_upper_bound']
            if exclusion and (memory>cap or row['upper']<=screening):
                residual.append(dict(memory=memory,upper=row['upper']))
                continue
            identity=dict(Q=q,ED=ed,B=B,test_t=list(map(str,ts)),source_sha256=hashlib.sha256(Path(base['output']).read_bytes()).hexdigest(),pruning_target=screening)
            tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            cached=task/(tag+'.json')
            if cached.exists():rr=json.loads(cached.read_bytes())
            else:
                state,y=prepare(q,q,[(t,t) for t in ts],ed,B)
                conc=np.minimum(1,np.maximum(0,iv.up(2*state[1].max(-1)-1)))
                values=np.minimum(y[1],iv.up(y[1]*iv.up(conc*conc)))
                goal=math.nextafter((screening if exclusion else lows[f'M{memory}'])+2e-9,math.inf)
                if values.min()>goal:
                    target=np.maximum(0,iv.down(goal/np.maximum(y[1],iv.TINY)))
                    _,bounds=best_bounds(state,kernels,physical,128,target)
                    values=np.minimum(values,iv.up(y[1]*bounds))
                rr=dict(**identity,memory=memory,upper=min(row['upper'],float(values.min())),test_state_upper=list(map(float,values)))
                save(cached,rr)
            residual.append(rr)
        bounds={}
        for m in range(1,9):
            key=f'M{m}'
            # All objects absent from the residual list were already bounded
            # by their own feasible lower plus the fresh engine tolerance.
            remaining=math.nextafter(lows[key]+report['tolerance'],math.inf)
            hi=max([remaining]+[x['upper'] for x in residual if x['memory']<=m])
            bounds[key]=min(r['upper'][key],hi)
        rounds.append(dict(B=r['B'],upper=bounds,residuals=len(residual)))
        print('XYZ REFINED',q,r['B'],bounds,flush=True)
    results={key:dict(lower=lows[key],upper=max(r['upper'][key] for r in rounds)) for key in lows}
    result=dict(complete=True,Q=q,results=results,rounds=rounds,seconds=time.monotonic()-start,
                fresh_base=str(base['output']),fresh_base_sha256=hashlib.sha256(Path(base['output']).read_bytes()).hexdigest(),
                scope=base['scope'],test_t=list(map(str,ts)),historical_upper_values_used=False)
    if exclusion:
        result.update(pruning_target=screening,exclusion_memory_cap=cap,
            strict=results['M5']['upper']<results['M6']['lower'],
            interpretation='Target-oriented complete registered-family exclusion; pruning target is not an achievable lower. Budgets above the exclusion cap retain their previous valid upper bounds.')
    save(outpath,result)
    return result


def block_separation(q):
    """Full block-class exclusion at a specified target, not an optimum.

    Any rigorous per-candidate upper already below the target can be left
    coarse. The target is NEVER a lower witness for the smaller memory.
    This avoids computing irrelevant entropy precision for a separation.
    """
    from fractions import Fraction as F
    basepath=HERE/'full_bb84/registered_refined'/str(q)/'receipt.json'
    alternative=HERE/'full_bb84/registered_exclusion'/str(q)/'receipt.json'
    if alternative.exists() and json.loads(alternative.read_bytes()).get('complete'):
        basepath=alternative
    registered=json.loads(basepath.read_bytes())
    choices=[]
    for m in range(2,9):
        L=F.from_float(registered['results'][f'M{m}']['lower'])
        U=F.from_float(registered['results'][f'M{m-1}']['upper'])
        if L>U:choices.append((L-U,m,L,U))
    if not choices:
        raise ValueError('No registered memory gap available at this point.')
    gap,m,L,U=max(choices)
    # A rational target strictly between the registered lower and upper.
    target=(L+U)/2
    task=HERE/'full_bb84/block_separation'/str(q)
    task.mkdir(parents=True,exist_ok=True)
    receipt=task/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_bytes()).get('complete'):
        return json.loads(receipt.read_bytes())
    source=REV/'code_extension/bb84_m5/audit_upper.py'
    code=source.read_text(encoding='utf-8')
    code=code.replace("job=HERE/'upper_runs'/tag;job.mkdir(parents=True,exist_ok=True)",
                      "job=RECHECK_OUTPUT/tag;job.mkdir(parents=True,exist_ok=True)")
    code=code.replace('lower_witness=float(lower[k])','pruning_target=float(lower[k])')
    # Only the lower memory is being excluded. Do not spend work on larger
    # block memories whose upper bounds cannot enter this comparison.
    # Include this restriction in the job identity to avoid cache collisions.
    code=code.replace('general_code_max=8,fixed_block_max=5,',
                      'general_code_max=8,fixed_block_max=5,memory_cap=RECHECK_MEMORY,')
    code=code.replace("entries=blocks.load_entries(args.permissions,b==p)",
                      "entries=[e for e in blocks.load_entries(args.permissions,b==p) if e['M']<=RECHECK_MEMORY]")
    ns=dict(__name__='block_separation_upper',__file__=str(source),RECHECK_OUTPUT=task,
            RECHECK_MEMORY=m-1)
    exec(compile(code,str(source),'exec'),ns)
    def target_source(*args,**kwargs):
        return [0.]+[float(target)]*8,['0','.125','.25','.5','1'],{}
    ns['lower_sources']=target_source
    args=argparse.Namespace(p=None,permissions='full',nmax=128,K=16,
                           tolerance=1e-11,quantum=1e-12,lower_file=[],test_t=[])
    r=ns['run'](args,q)
    output=task/r['job_fingerprint']/'result.json'
    # Preserve the numerical engine's bounds, but label the screening values
    # as targets in the saved evidence, never as feasible lower endpoints.
    r['pruning_targets']=r.pop('lower')
    r['scope']=f'Complete full-Clifford N<=5 flagged block upper at memory <= {m-1}, full AD menu and full-input cost. Numerical targets only control valid upper-bound pruning; no block lower or optimum is inferred. Rows above the memory cap are not full-class bounds at those budgets.'
    save(output,r)
    blockU=F.from_float(r['results'][f'M{m-1}']['upper'])
    result=dict(complete=r['complete'],Q=q,memory=m,high_lower=float(L),
                registered_low_upper=float(U),block_low_upper=float(blockU),
                union_low_upper=float(max(U,blockU)),strict=bool(max(U,blockU)<L),
                pruning_target_exact=str(target),results=r['results'],
                pruning_target_binary64_exact=str(F.from_float(float(target))),
                memory_cap=m-1,
                output=str(output),fresh_registered_sha256=hashlib.sha256(basepath.read_bytes()).hexdigest(),
                scope=r['scope'])
    save(receipt,result)
    print('COMPLETE CLIFFORD SEPARATION',q,m,result['strict'],result['union_low_upper'],float(L),flush=True)
    return result


def block_refine(q, high_noise=False):
    """Additional known-source upper tests for a complete exclusion run."""
    previous=HERE/'full_bb84/block_separation'/str(q)/'receipt.json'
    base=json.loads(previous.read_bytes())
    assert base['complete']
    target_override=None
    if high_noise:
        witness_file=HERE/'full_bb84/high_noise_witnesses'/str(q)/'receipt.json'
        witnesses=json.loads(witness_file.read_bytes());assert witnesses['complete']
        witness=witnesses['best'];m=witness['ED']['memory_upper_bound']
        reg_file=HERE/'full_bb84/registered_exclusion'/str(q)/'receipt.json'
        reg=json.loads(reg_file.read_bytes());assert reg['complete']
        L=witness['lower'];U=reg['results'][f'M{m-1}']['upper'];assert L>U
        target_override=(L+U)/2
        base={**base,'memory':m,'high_lower':L,'registered_low_upper':U,
              'witness':witness,'witness_sha256':hashlib.sha256(witness_file.read_bytes()).hexdigest(),
              'fresh_registered_sha256':hashlib.sha256(reg_file.read_bytes()).hexdigest()}
    task=HERE/'full_bb84'/('block_high_noise_refined' if high_noise else 'block_refined')/str(q)
    task.mkdir(parents=True,exist_ok=True)
    receipt=task/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_bytes()).get('complete'):
        return json.loads(receipt.read_bytes())
    source=REV/'reporting/core_refinement/refine_pointwise.py'
    code=source.read_text(encoding='utf-8')
    code=code.replace("assert base['complete'] and base['permissions']=='real'",
                      "assert base['complete'] and base['permissions']=='full'")
    code=code.replace("base['lower']","base['pruning_targets']")
    code=code.replace("blocks.load_entries('real',base['b']==base['p'])",
                      "blocks.load_entries('full',base['b']==base['p']) if e['id'] in bounds")
    code=code.replace("job=HERE/'runs'/fingerprint;job.mkdir(parents=True,exist_ok=True)",
                      "job=RECHECK_OUTPUT/fingerprint;job.mkdir(parents=True,exist_ok=True)")
    code=code.replace("permissions='real'","permissions='full'")
    code=code.replace('lower_witness=lower','pruning_target=lower')
    if high_noise:
        code=code.replace("base=json.loads(args.base.read_text())",
                          "base=json.loads(args.base.read_text());base['pruning_targets']=[0.]+[RECHECK_TARGET]*8")
        # Target-oriented exclusion needs valid bounds, not machine-precision
        # optimization of every per-flag action already safely below target.
        code=code.replace('target=np.maximum(best[ids]/np.maximum(factor[1][ids],iv.TINY),localtol)',
                          "target=np.maximum(best[ids]/np.maximum(factor[1][ids],iv.TINY),np.maximum(localtol,iv.down(goal(bounds[identifier])/np.maximum(importance,1e-300))))")
    ns=dict(__name__='full_block_refinement',__file__=str(source),RECHECK_OUTPUT=task,
            RECHECK_TARGET=target_override)
    exec(compile(code,str(source),'exec'),ns)
    grid=['0','.03125','.0625','.09375','.125','.15','.175','.1875','.2','.2125','.225','.2375',
          '.25','.275','.3','.375','.5','.75','1']
    if high_noise:
        # Any legal source is an upper relaxation of the minimax problem.
        # A compact first pass avoids evaluating many unnecessary entropy
        # boxes. The grid is committed to the job fingerprint and receipt.
        grid=['0','.09375','.25']
    sys.argv=[str(source),'--base',base['output'],'--relative','0','--absolute','1e-11',
              '--test-t',*grid]
    ns['main']()
    files=list(task.glob('*/result.json'));assert len(files)==1
    path=files[0];result=json.loads(path.read_bytes())
    result['scope']=base['scope']+' Additional outward legal-source tests sharpen residual block uppers; every base instrument is retained.'
    save(path,result)
    m=base['memory'];block=result['results'][f'M{m-1}']['upper']
    union=max(block,base['registered_low_upper'])
    out={**base,'output':str(path),'block_low_upper':block,'union_low_upper':union,
         'strict':union<base['high_lower'],'fresh_block_base_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
         'results':result['results'],'test_t':grid,'scope':result['scope']}
    if high_noise:out['pruning_target']=target_override
    save(receipt,out)
    print('REFINED CLIFFORD SEPARATION',q,m,out['strict'],union,out['high_lower'],flush=True)
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--q',type=float,required=True)
    parser.add_argument('--stage',choices=['baseline','block-upper','block-compact','registered-upper','registered-refine','registered-separation','block-separation','block-refine','block-high-noise'],required=True)
    parser.add_argument('--watanabe-only',action='store_true')
    args=parser.parse_args()
    if args.stage=='baseline':baseline(args.q,not args.watanabe_only)
    elif args.stage=='block-upper':block_upper(args.q)
    elif args.stage=='block-compact':block_upper(args.q,True)
    elif args.stage=='registered-upper':registered_upper(args.q)
    elif args.stage=='registered-refine':registered_refine(args.q)
    elif args.stage=='registered-separation':registered_refine(args.q,True)
    elif args.stage=='block-separation':block_separation(args.q)
    elif args.stage=='block-refine':block_refine(args.q)
    else:block_refine(args.q,True)
