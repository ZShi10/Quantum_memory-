"""Complete small-block controls needed for the multi-error union comparison."""
from pathlib import Path
import json,subprocess,sys
HERE=Path(__file__).resolve().parent
for q,ceiling in [('.19','.01338'),('.21','.003486'),('.22','.0013507')]:
    output=HERE/'bb84'/str(float(q))/'block_component/result.json'
    if output.exists() and json.loads(output.read_bytes()).get('complete'):continue
    args=[sys.executable,'-X','utf8',str(HERE/'run_numerical.py'),'block_component_upper.py','--q',q,'--ceiling',ceiling]
    with (HERE/'block_components.log').open('ab') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError((q,result.returncode))
    print('BLOCK COMPONENT COMPLETE',q,flush=True)
