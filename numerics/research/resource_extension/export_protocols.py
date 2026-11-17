"""Export explicit measurement recipes and classical acceptance rules."""
from common import *
from collections import deque
from fractions import Fraction as F
import json
from circuits import apply_gate
from rates import CodeKernel
from kernels import load


def read(path):return json.loads(path.read_text())


def input_axis_gates(permutation):
    """Choose U with U^dagger P_a U = +/- P_permutation[a]."""
    paulis=[0,1,3,2];queue=deque([((0,1,2,3),[]) ]);seen=set()
    while queue:
        mapping,gates=queue.popleft()
        if all(mapping[permutation[a]]==a for a in range(4)):return gates
        if mapping in seen:continue
        seen.add(mapping)
        for op in ('h','s'):
            nxt=tuple(paulis.index(apply_gate(paulis[j],(op,0),1)) for j in mapping)
            queue.append((nxt,gates+[op]))
    raise AssertionError(permutation)


def main():
    combined=read(DATA/'combined_results.json');memory=read(DATA/'memory_refinement.json')
    small={e['id']:e for e in read(OLD/'data/blocks.json')['entries']}
    graphs={e['id']:e for e in read(OLD/'data/graph_family_blocks.json')['entries']}
    graphs.update({e['id']:e for e in read(DATA/'graphs.json')['entries']})
    codes={e['id']:e for e in read(OLD/'data/codes.json')['entries']}
    codes.update({e['id']:e for e in read(DATA/'codes.json')['entries']})
    bbdefs={}
    for path in sorted(DATA.glob('bb84_[0-9]*.json')):
        bbdefs.update(read(path)['protocol_library'])
    edids=set();codeids=set()
    for r in combined['protocols'].values():
        edids.update(r.get('EDs',[r.get('ED')]))
        codeids.update(b['code'] for b in r.get('branches',[]) if b.get('code'))
        for cell in r.get('policy',[]):
            codeids.update(b['code'] for b in cell.get('processing',[]))
    edids.discard(None)
    definitions={}
    for sid in sorted(edids):
        if sid in small:
            e=small[sid]
            definitions[sid]={k:v for k,v in e.items() if k not in ('tensor','multiplicity')}
            definitions[sid]['full_terminal_implementation']=memory['small_full'][sid]
            definitions[sid]['note']='Original checks and logical observables use x + (z << m) binary Pauli encoding. Full-terminal masks give key then check bits directly; relative Alice/Bob offsets cancel. Original compiled_frame applies to the original compiled ED gates, not to the replacement key/check masks.'
        elif sid in graphs or sid in bbdefs:
            extra=bbdefs.get(sid,{})
            gid=extra.get('graph_id',sid);e=graphs[gid]
            definitions[sid]={k:v for k,v in e.items() if k not in ('characters','werner_weights')}
            definitions[sid]['full_terminal_implementation']=memory['results'][gid]
            perm=extra.get('axis_permutation',[0,1,2,3])
            definitions[sid]['input_axis_permutation']=perm
            definitions[sid]['input_axis_gates_on_each_wire']=input_axis_gates(perm)
            definitions[sid]['key_pivot']=e['logical_dual_vector'].bit_length()-1
        elif sid.startswith('restart_pump_'):
            count=int(sid.rsplit('_',1)[1])
            definitions[sid]=dict(family='raw-pair restart pumping',r=count,M=1 if count==1 else 2,
                procedure='Keep the control pair after each successful bilateral B step with a fresh raw target; a failed comparison discards the control and restarts. After r raw pairs on the success path, apply bilateral H, measure Z, and use the saved AD rule. The rate uses the full expected restart cost.')
        else:
            assert sid.startswith(('direct_','pump_')),sid
            definitions[sid]=dict(family='pilot-production direct or pumping',
                source_file='research/joint_optimization/streaming.py',
                procedure='Use parameters from each policy cell. direct axis 0/1/2 has Pauli permutation [0,1,2,3]/[0,3,2,1]/[0,1,3,2]. pump axis 0 checks Z parity then uses complementary X key measurement; axis 1 swaps X and Z. Restart after any failed raw-pair pump step; r counts the success path.')
    kernels={k.entry['id']:k for k in load()}
    code_defs={}
    for cid in sorted(codeids):
        e=codes[cid];k=kernels.get(cid)
        if k is None:k=CodeKernel(e)
        code_defs[cid]={key:e[key] for key in ('n','k','d','rows')}
        code_defs[cid]['relative_bit_cosets_by_class']=k.syndrome_classes
    for r in combined['protocols'].values():
        path=ROOT/r['source'] if r['source'].startswith('../') else DATA/r['source']
        assert path.exists(),path
        r['source_file']=path.resolve().relative_to(ROOT.parents[1]).as_posix()
        for cell in r.get('policy',[]):
            for branch in cell.get('processing',[]):
                classes=code_defs[branch['code']]['relative_bit_cosets_by_class']
                branch['kept_relative_bit_coset_representatives']=sorted(x for j in branch['keep_classes'] for x in classes[j])
    result=dict(conventions=dict(memory='Peak local live qubits per party, including the input port and legal pilot stage; full and real permissions differ.',
        BB84_parameter='Every saved interval is for t=z/min(eb,ep), from 0 to 1. Intervals are decision cells in the unknown-state parameter, not physical-Q switching intervals.',
        graph_key='Recover graph-basis bits y from the recorded masks. Let pivot be the highest set bit of v. Key is y[pivot]; check j is y[j] XOR v[j]*y[pivot], for j != pivot in increasing wire order. New XG protocols accept zero relative ED syndrome; old G records retain the listed relative syndromes.',
        graph_gate_order='Apply input-axis gates, original per-wire H if input_rotations requests it, then the replacement full graph-measurement gates. Bob uses the complex-conjugate circuit. Streaming cuts specify when wires may be measured and reloaded; classical terminal operations remain classical.',
        code_bits='Integers encode binary vectors with wire 0 in the least significant bit. rows are generator rows. A bit-error coset representative is min(x XOR c for c in C). SPC [n,n-1,2] can use rows (1<<i) XOR (1<<(n-1)), i=0..n-2; repetition uses the all-one row.',
        AD_acceptance='BB84 short-code rules list the exact kept cosets. Werner short-code/SPC rules keep every public syndrome with positive conditional hashing gain for that known input. A final repetition rule retains zero syndrome. B^k means k pooled binary AD rounds; it is not billed as one fixed block of length 2^k.',
        pilot='Full permissions can estimate z using a negligible asymptotic Y-test sample. Real permissions require the two-register Bell-basis pilot. Rates cover all compatible inputs after this predeclared decision rule; no finite-key certification is claimed.'),
        ed_definitions=definitions,code_definitions=code_defs,protocols=combined['protocols'])
    (DATA/'protocol_catalog.json').write_text(json.dumps(result,separators=(',',':')))
    lines=['# 可重建协议目录','',
        '完整数据在 `data/protocol_catalog.json`：各扫描点/预算的方案、量子门和调度、经典码生成矩阵、BB84 的完整反馈规则及保留的误差陪集均已展开。源文件路径从工作区根目录解析。', '',
        '这里的最佳是已保存构造中的最佳。memory 是每方峰值活跃 qubit 数，包含输入端口；BB84 real 的 pilot 需要两个寄存器。', '',
        '## Werner：代表性协议','']
    for q,M in ((.28,2),(.28,8),(.30,8),(.305,8),(.31,8),(.315,8)):
        r=combined['protocols'][f'SIX_{q:.5f}_M{M}'];e=definitions[r['ED']]
        lines += [f"### Q={100*q:g}%，预算 M<={M}",'',f"产率 {r['rate']:.10e} bits/raw pair；{r['ED']}，总输入 m={r['m']}，实现 M={r['M']}。AD：{r['AD_summary']}。",'']
        if 'edges' in e:
            f=e['full_terminal_implementation']
            lines += [f"原始图边：`{e['edges']}`。",'',f"逻辑 dual vector：`{e['logical_dual_vector']}`；原始局部 H 选择：`{e.get('input_rotations')}`。",'',
                f"full 编译的局部互补顶点序列：`{f['local_complementation_vertices']}`；输入顺序：`{f['input_order']}`。完整门与输出重标记见目录。",'']
        elif r['ED'].startswith('E'):
            lines += [f"Pauli 编码 x+(z<<m)：checks=`{e['checks']}`，logical X=`{e['logical_X']}`，logical Z=`{e['logical_Z']}`。",'',
                f"full 终端门序列：`{e['full_terminal_implementation']['gates']}`。",'']
        else:lines += ['采用 raw-pair restart pumping；每次失败都重新开始，计率使用平均原始纠缠对成本。','']
    lines += ['## BB84：23% 与 24% 的 full 反馈规则','',
        '下表 t=z/min(eb,ep)，并非 z 本身。pilot 后按 t 选择生产协议。相邻且处理规则相同的认证单元合并显示；原始单元和各自严格下界完整保留在 JSON。所有未列出的 ED 分支或 AD 陪集均丢弃。','']
    for q in (.23,.24):
        r=combined['protocols'][f'BB_{q:.5f}_{q:.5f}_full_M8'];merged=[]
        for c in r['policy']:
            key=json.dumps([c['ED'],c.get('processing')],sort_keys=True)
            if merged and merged[-1]['key']==key and F(merged[-1]['b'])==F(c['interval'][0]):merged[-1]['b']=c['interval'][1]
            else:merged.append(dict(key=key,a=c['interval'][0],b=c['interval'][1],cell=c))
        lines += [f"### eb=ep={100*q:g}%：保证率 {r['rate']:.10e}，实现 M={r['M']}",'',
            '| t 区间 | ED | 经典处理：ED syndrome → 码；保留陪集代表 |','|---|---|---|']
        for part in merged:
            c=part['cell'];processing=[]
            for b in c.get('processing',[]):
                co=code_defs[b['code']]
                processing.append(f"{b['syndrome']} → [{co['n']},{co['k']},{co['d']}] {b['code']}; {b['kept_relative_bit_coset_representatives']}")
            lines.append(f"| [{part['a']}, {part['b']}] | {c['ED']} | {'; '.join(processing)} |")
        lines += ['']
    lines += ['固定 (eb,ep) 的规则覆盖整个兼容态区间。沿连续物理误差 Q 改变时，最佳协议的切换位置尚未认证。']
    (ROOT/'PROTOCOL_CATALOG.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(protocols=len(result['protocols']),ED_definitions=len(definitions),code_definitions=len(code_defs))))


if __name__=='__main__':main()
