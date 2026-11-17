"""Export a compact, reproducible full structural catalog, with exact DAG IDs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from resource_model.model import *
from resource_model.catalog import *
import argparse
import gzip
import hashlib
import json


def export(max_leaves=5, max_checks=7):
    memberships = {}
    for layer in ordered_tree_layers(max_leaves).values():
        for tree in layer:
            memberships.setdefault(tree, []).append('complete_ordered_tree_leaf_bound')
    for word, tree in variable_pumps(max_checks):
        memberships.setdefault(tree, []).append('variable_decoder_raw_pump')
    nodes = {}
    for tree in memberships:
        item = dict(kind='raw') if isinstance(tree, Raw) else dict(kind='node', decoder=tree.decoder,
                        left=tree_id(tree.left), right=tree_id(tree.right))
        nodes[tree_id(tree)] = dict(**item, retained_memory=retained_memory(tree),
                                   successful_ancestry_inputs=ancestry(tree), depth=depth(tree))
    records = [dict(tree_id=tree_id(tree), final_axis=axis,
                    terminal_memory=terminal_memory(tree, axis),
                    implementation=terminal_plan(tree, axis)[1], family_memberships=memberships[tree])
               for tree in memberships for axis in AXIS_PERMUTATIONS]
    out = dict(schema_version=1, coverage=coverage(max_leaves, max_checks),
               family='explicit_binary_tree', retry_rule=RETRY_RULE,
               memory_proof_kind='exact_in_registered_depth_first_and_terminal_reduction_schedule_class',
               global_minimality_proved=False,
               note='Every node ID hashes its full ordered labeled tree. Child-ID DAG is lossless. Y rows require enlarged_axes permission.',
               nodes=nodes, terminal_candidates=records)
    encoded = json.dumps(out, separators=(',', ':'), sort_keys=True).encode('utf-8')
    path = ROOT/'catalog.json.gz'
    path.write_bytes(gzip.compress(encoded, mtime=0))
    # Independent reconstruction of each DAG node checks references and hashes.
    rebuilt = {}
    def rebuild(key):
        if key not in rebuilt:
            item = nodes[key]
            tree = RAW if item['kind']=='raw' else Node(item['decoder'], rebuild(item['left']), rebuild(item['right']))
            assert tree_id(tree) == key
            rebuilt[key] = tree
        return rebuilt[key]
    for key in nodes:
        rebuild(key)
    assert len(nodes) == out['coverage']['unique_structural_trees']
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    manifest = dict(status='complete', compressed_catalog=path.name, compressed_bytes=path.stat().st_size,
                    compressed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    uncompressed_sha256=hashlib.sha256(encoded).hexdigest(),
                    nodes=len(nodes), terminal_candidates=len(records), source_sha256=sources,
                    coverage=out['coverage'])
    (ROOT/'catalog_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_sha256','coverage')}, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-leaves', type=int, default=5)
    parser.add_argument('--max-pump-checks', type=int, default=7)
    args=parser.parse_args()
    export(args.max_leaves,args.max_pump_checks)
