"""Check content-addressed QKD evidence after unpacking the supplied 7z archive.

Usage:
  python audit_archive.py --manifest PATH/manifest.json --archive evidence.7z
No scientific optimum is independently replayed by this integrity checker.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,default=Path('evidence_integrity_audit.json'))
    args=parser.parse_args()
    manifest=args.manifest.resolve()
    data=json.loads(manifest.read_text())
    root=manifest.parent
    checked={};failures=[]
    for logical,record in data['files'].items():
        payload=(root/record['payload']).resolve()
        if not payload.is_relative_to(root):
            raise ValueError(f'Unsafe payload path: {payload}')
        if payload not in checked:
            checked[payload]=(sha256_file(payload),payload.stat().st_size)
        sha,size=checked[payload]
        if sha != record['sha256'] or size != record['size']:
            failures.append(logical)
    result={'archive_sha256':sha256_file(args.archive) if args.archive else None,
            'manifest_status':data.get('status'),'published':data.get('published'),
            'release_eligible':data.get('release_eligible'),
            'path_records':len(data['files']),'unique_payloads':len(checked),
            'hash_or_size_failures':failures,
            'point_status_counts':data.get('point_status_counts'),
            'unresolved_references':data.get('unresolved_references'),
            'independent_claim':'Listed hashes and sizes only; no replay of scientific optimum certificates.'}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
