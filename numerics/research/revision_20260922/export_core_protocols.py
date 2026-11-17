"""Export executable complete-block winners, with signed streaming records."""
from core import *
import hashlib


def main():
    library_path=COMPLETE/'data/werner_complete_library.json'
    library=json.loads(library_path.read_text())
    entries={e['id']:e for e in library['entries']}
    source=json.loads((DATA/'werner_complete_joint_n128.json').read_text())
    cert=json.loads((DATA/'werner_joint_union_core_certified.json').read_text())
    rows=[];definitions={}
    additional_path=DATA/'fixed_block_additional_certificates.json'
    additional=json.loads(additional_path.read_text()) if additional_path.exists() else dict(results={})
    replay_kernels=None
    codes={e['id']:e for e in json.loads((EXT/'data/codes.json').read_text())['entries']}
    for item in source['results']:
        q=item['Q'];proof=cert['results'].get(f'{q:.12g}',{})
        for budget,row in item['results'].items():
            name=row['ED'];entry=entries[name];graph=library['graphs'][entry['graph']]
            if name not in definitions:
                v=entry['logical_dual_vector'];pivot=(v&-v).bit_length()-1
                checks=[(1<<i)^(((v>>i)&1)<<pivot) for i in range(entry['m']) if i!=pivot]
                definitions[name]=dict(m=entry['m'],M=entry['M'],graph=graph,
                   key_original_generator_index=pivot,public_check_generator_masks=checks,
                   public_bit_order='Increasing original generator index, omitting the key pivot; least significant syndrome bit is first.',
                   transcript_rule='Keep intermediate compiler outcomes private; reconstruct original outcomes using the affine map; announce original public checks only.',
                   branches=entry['branches'])
            target=sorted((v['state'],json.dumps(v['AD'],sort_keys=True)) for v in row['branches'])
            witness=next((w for w in proof.get('core_witnesses',[]) if w['ED']==name and
                          sorted((v['state'],json.dumps(v['AD'],sort_keys=True)) for v in w['branches'])==target),None)
            if witness is None:
                # The long-code search may finish after a core upper job
                # has started. Its new selected lower witness then needs
                # its own replay; an older same-ID witness is insufficient.
                key=f'{q:.12g}_'+name+'_'+json.dumps(target,sort_keys=True)
                if key not in additional['results']:
                    from certify_werner_joint_union import raw_bounds,replay
                    from bb84_fixed import physical_kernel
                    import stable_intervals as iv
                    if replay_kernels is None:
                        replay_kernels={k.entry['id']:k for k in load_codes()}
                        physical={key:physical_kernel(k) for key,k in replay_kernels.items()}
                    m=row['m'];weights=np.load(COMPLETE/f'data/werner_states_m{m}.npz')['weights']
                    raw=raw_bounds(q,{m:weights});mass=iv.positive_sum(raw)
                    additional['results'][key]=replay(row,raw,mass,{m:0},replay_kernels,physical)
                    atomic_json(additional_path,additional)
                witness=dict(additional['results'][key],source='fixed_block_additional_certificates.json',source_key=key)
            short={a['AD']['code']:codes[a['AD']['code']] for a in row['branches'] if a['AD']['kind']=='linear'}
            rows.append(dict(Q=q,budget=budget,ED=name,rate_numerical=row['rate'],
                             certificate=witness,kept_branches=row['branches'],short_code_definitions=short,
                             discarded_syndromes='All public syndromes not listed among kept_branches.'))
    atomic_json(DATA/'fixed_block_protocols.json',dict(protocols=rows,definitions=definitions,
          source_library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
          scope='Fixed-input complete terminal blocks; all m inputs consumed. A row is an executable lower witness, not a whole-union upper bound.'))
    e=definitions['LC8_175_v12']
    print('Fixed block protocols:',len(rows),'definitions:',len(definitions),flush=True)
    print('Q=.30 M3 example:',dict(edges=e['graph']['edges'],order=e['graph']['compilation']['input_order'],
          key=e['key_original_generator_index'],checks=e['public_check_generator_masks']),flush=True)


if __name__=='__main__':main()
