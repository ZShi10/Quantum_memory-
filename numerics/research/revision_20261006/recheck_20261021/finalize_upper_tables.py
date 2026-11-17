"""Publish fresh endpoints only after all original upper claims pass."""
from pathlib import Path
from fractions import Fraction as F
import csv, hashlib, json, math
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
OUT=PROJECT/'revised/data'


def main():
    report=json.loads((HERE/'comparison.json').read_bytes())
    assert report['status']=='complete' and report['completed_noise_points']==77
    assert report['fresh_upper_rows']==616 and not report['gap_issues']
    assert len(report['fresh_target_windows'])==5
    points={(r['model'],r['Q'],r['memory']):r for r in report['rows']}
    path=OUT/'global_rate_landscape.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    changed=0
    for row in rows:
        key=(row['source_model'],str(float(row['Q'])),int(row['memory_budget']))
        if key in points:
            row['upper']=repr(points[key]['fresh_upper'])
            row['reproduction_status']='whole-family upper freshly recomputed; declared optimization gap checked'
            changed+=1
    assert changed==616
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (OUT/'fresh_upper_enclosures.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report['rows'][0]));w.writeheader();w.writerows(report['rows'])
    (OUT/'fresh_upper_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    # The target-window CSV schema is preserved; only the upper endpoints
    # are replaced by exact-rationally assembled, outward fresh values.
    path=OUT/'figure4_target_windows.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        item=next(r for r in report['fresh_target_windows'] if r['Q']==row['Q'] and r['memory']==int(row['memory']))
        row['upper_low']=repr(item['fresh_extended_upper'])
        row['upper_low_exact']=item['fresh_extended_upper_exact']
        ratio=F(row['lower_high_exact'])/F(row['upper_low_exact'])
        row['ratio_lower_exact']=str(ratio)
        value=float(ratio)
        if F.from_float(value)>ratio:value=math.nextafter(value,-math.inf)
        row['ratio_lower']=repr(value)
        row['upper_source']='fresh finite upper + proved enlargement; exact rational arithmetic'
        row['upper_active_term']=item['active_upper_term']
    # The existing table prints conservative upward-rounded endpoints.
    # Verify rather than silently retaining potentially stale display values.
    printed=['0.00043613942','0.00054335049','0.0000076293946','0.000064014729','0.000092512415']
    assert all(F(row['upper_low_exact'])<=F(value) for row,value in zip(rows,printed))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    result=dict(status='passed',upper_points=77,upper_rows=616,
        BB84_points=71,Werner_anchors=6,excluded_Werner_lower_only_points=114,
        target_windows_passed=5,historical_upper_endpoints_used_as_proof=False,
        maximum_fresh_minus_old=max(r['difference'] for r in report['rows']),
        maximum_absolute_endpoint_change=max(abs(r['difference']) for r in report['rows']),
        comparison_sha256=hashlib.sha256((HERE/'comparison.json').read_bytes()).hexdigest())
    (PROJECT/'revised/validation/fresh_upper_campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
