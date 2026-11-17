"""Isolated reuse of the audited numerical engines, with explicit providers.

Call configure once per process, BEFORE importing any legacy rate/search module.
The frozen algebra and complete-block data remain read-only dependencies. New
mutable kernels, interval heaps and search records live under the supplied run.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
LEGACY = HERE.parent / 'revision_20260922'


def configure(destination, code_provider=None):
    if 'core' in sys.modules:
        raise RuntimeError('configure must precede importing legacy core or engines')
    destination = Path(destination).resolve()
    if not destination.is_relative_to(HERE.resolve()):
        raise ValueError('mutable run must stay inside revision_20261006')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(LEGACY))
    import core
    core.ROOT = destination
    core.DATA = destination / 'data'
    core.DATA.mkdir(parents=True, exist_ok=True)
    original_loader = core.load_codes
    if code_provider is not None:
        core.load_codes = code_provider
    worker = LEGACY / 'fixed_policy_lp.py'
    shutil.copy2(worker, destination / worker.name)
    return core, original_loader


def fingerprint(extra_files=()):
    # The older algebra, compiler, and bounds are executable dependencies too.
    # A cache keyed only to the latest wrapper would silently survive changes
    # to those sources.
    files = set()
    for folder in (LEGACY, HERE.parent / 'joint_optimization',
                   HERE.parent / 'completion', HERE.parent / 'resource_extension'):
        files.update(folder.glob('*.py'))
    files.update((HERE / 'protocol_spec.json', Path(__file__).resolve()))
    files.update(Path(p).resolve() for p in extra_files)
    files = sorted(files)
    hashes = {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(digest=digest, source_sha256=hashes)
