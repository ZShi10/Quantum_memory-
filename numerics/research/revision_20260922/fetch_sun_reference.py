"""Freeze the author's published repository inputs. Never execute remote code."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = '176a9cf7731b93fb2639c59ccc692b8be46915c0'
REPO = 'zinc-sun/ED-AD'
DEST = ROOT / 'sources' / 'sun'


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'QKD-research-reproduction'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    url = f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'
    tree = json.loads(download(url))
    assert tree['sha'] == COMMIT and not tree.get('truncated')
    (DEST / 'repository_tree.json').write_text(json.dumps(tree, indent=2), encoding='utf-8')
    entries = {e['path']: e for e in tree['tree'] if e['type'] == 'blob'}
    paths = ['README.md', 'visualizer_werner.ipynb']
    paths += sorted(p for p in entries if p.startswith('werner/') and p.endswith('.ipynb'))
    paths += sorted(p for p in entries if p.startswith('data/werner/bb84_key_rate_') and p.endswith('.npy'))
    records = []
    for name in paths:
        entry = entries[name]
        assert entry['size'] < 64 * 1024 * 1024
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = target.read_bytes() if target.exists() else b''
        if blob_sha(data) != entry['sha']:
            data = download(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}')
            assert len(data) == entry['size'] and blob_sha(data) == entry['sha'], name
            target.write_bytes(data)
        records.append(dict(path=name, bytes=len(data), git_blob_sha1=entry['sha'],
                            sha256=hashlib.sha256(data).hexdigest()))
        print('Verified author file', name, len(data), flush=True)
    manifest = dict(repository=f'https://github.com/{REPO}', commit=COMMIT,
                    files=records, status='downloaded_and_hash_verified',
                    missing_author_rate_arrays=[p for p in ['data/werner/bb84_key_rate_42.npy'] if p not in entries],
                    scope='Author data and source are preserved verbatim. No notebook executed. No figure-reproduction claim yet.')
    (DEST / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
