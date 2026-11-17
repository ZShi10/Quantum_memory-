"""Copy approved PyPI dependencies into the ordinary workspace ACL hierarchy.

pip moved directories from a protected Windows temporary tree. Copying file
contents creates ordinary workspace files; no ACL or sandbox rule is changed.
This optional local convenience is excluded from the distributable source.
"""
from pathlib import Path
import hashlib
import json
import shutil
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'.deps';target=ROOT/'.runtime'
target.mkdir(exist_ok=True)
assert source.parent==target.parent==ROOT
count=0
for path in source.rglob('*'):
    out=target/path.relative_to(source)
    if path.is_dir():out.mkdir(exist_ok=True)
    elif path.is_file():
        shutil.copyfile(path,out)
        assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(out.read_bytes()).digest()
        count+=1
(target/'READY.json').write_text(json.dumps(dict(status='verified_copy',files=count))+'\n')
print('Dependency contents copied and verified:',count)
