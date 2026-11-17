"""Copy identified research evidence into the new release, with hash checks.

This development-only importer never modifies source files. The release
already contains its outputs, so portable reproduction does not run it.
"""
from pathlib import Path
import hashlib,json,shutil,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
SHA='6894ed43294fc2cecb41b004bcea0fb138751929a3ee8bdeb0deaeb174f443f4'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=PROJECT/'research/revision_20261006/release_tools/release_ready_bundles'/SHA
    manifest=source/'manifest.json';assert digest(manifest)==SHA
    dest=ROOT/'evidence/legacy_frozen';dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(manifest,dest/'manifest.json')
    doc=json.loads(manifest.read_bytes());copied=set()
    for r in doc['files'].values():
        relative=r['payload']
        if relative in copied:continue
        p=source/relative;assert digest(p)==r['sha256']
        out=dest/relative;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        assert digest(out)==r['sha256'];copied.add(relative)
    print('Copied and verified',len(copied),'frozen payloads',flush=True)
    literature=ROOT/'data/literature_inputs';literature.mkdir(parents=True,exist_ok=True)
    library=PROJECT/'manuscript/data/small_clifford_library.json'
    shutil.copyfile(library,literature/'small_clifford_library.json')
    author=PROJECT/'research/revision_20260922/sources/sun/data/werner'
    names=['2','22','23','24','3','32','33','4'];curves={};sources=[]
    for name in names:
        p=author/f'bb84_key_rate_{name}.npy';a=np.load(p,allow_pickle=False)
        curves[name]=a.reshape(500,-1).max(axis=1)
        sources.append(dict(size=name,filename=p.name,sha256=digest(p),shape=list(a.shape),
                            transformation='reshape(500,-1).max(axis=1); original author grid F=linspace(0.5,1,500)'))
    with (literature/'sun_author_envelopes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['F','Q']+names)
        for i,F in enumerate(np.linspace(.5,1,500)):w.writerow([F,2*(1-F)/3]+[curves[n][i] for n in names])
    provenance=dict(author_commit='176a9cf7731b93fb2639c59ccc692b8be46915c0',sources=sources,
                     weight_library_sha256=digest(library),
                     note='Author matrices are preserved in the original workspace; the release includes their exact per-grid envelopes and original hashes. No digitization or rescaling.')
    (literature/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Frozen author envelopes and independent Clifford weight library',flush=True)
if __name__=='__main__':main()
