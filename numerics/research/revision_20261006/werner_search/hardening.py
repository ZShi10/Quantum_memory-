"""Explicit input/dependency and finite-class witness checks."""
from pathlib import Path
from collections import Counter


def dependency_files(revision):
    project=revision.parent.parent
    files=[revision/'bb84_search/n8_adapter.py']
    files+=list((revision/'code_extension/data/kernels').glob('*.npz'))
    files+=list((project/'research/resource_extension/data/kernels').glob('*.npz'))
    files+=[project/'research/completion/data/werner_complete_library.json']
    files+=list((project/'research/completion/data').glob('werner_states_m*.npz'))
    return files


def check_action(action,kernels):
    if action is None:return
    B=action['classical_B_rounds']
    assert isinstance(B,int) and 0<=B<=16
    kind=action['kind'];n=action['n'];mode=action['mode']
    if kind=='Watanabe_2007':
        assert n==2 and mode=='no_otp_hashed'
        return
    assert mode in ('no_otp','otp_plain','otp_compressed')
    if kind=='linear':assert action['code'] in kernels and n==kernels[action['code']].n
    else:assert kind in ('SPC','repetition') and isinstance(n,int) and 8<=n<=128


def check_record(record,budget):
    from resource_model.model import (structural_record,homogeneous_tree,pumping_tree,
                                     tree_from_dict,canonical)
    from resource_model.catalog import structural_catalog
    assert record['final_axis'] in ('X','Y','Z')
    family=record['family']
    if family=='real_decoder_recurrence':
        assert len(record['path'])<=7 and set(record['path'])<=set('ZXRC')
        tree=homogeneous_tree(record['path'])
    elif family=='real_decoder_pumping':
        assert record['primitive'] in 'ZXRC' and len(record['primitive'])==1
        assert isinstance(record['r'],int) and 1<=record['r']<=160
        tree=pumping_tree(record['primitive']*(record['r']-1))
    else:
        assert family=='explicit_binary_tree'
        tree=tree_from_dict(record['tree'])
        members={t for t,_ in structural_catalog(5,7)}
        assert tree in members
    regenerated=structural_record(tree,record['final_axis'],'enlarged_axes')
    assert regenerated['memory_upper_bound']==record['memory_upper_bound']<=int(budget[1:])
    return regenerated


def check_block_rows(block,data,kernels):
    entries={e['id']:e for e in data['entries']}
    for budget,row in block['results'].items():
        e=entries[row['ED']]
        assert row['m']==e['m'] and row['M']==e['M']<=int(budget[1:])
        available=Counter((b['state'],b['factor']) for b in e['branches'])
        for branch in row['branches']:
            key=branch['state'],branch['factor']
            assert available[key]>0
            available[key]-=1
            check_action(branch['AD'],kernels)
