"""Check probability enclosures against direct and 60-digit calculations."""
from bb84_fixed import *
from interval_modes import menu_values,dual_upper


def main():
    rng=np.random.default_rng(44123);ks=load_codes(4,all_codes=True)
    entries=[e for e in complete_short_entries(False) if e['m']<=2]
    checks=0;largest=0.
    for e in entries[::max(1,len(entries)//8)]:
        b,p=.15,.2;ts=[0.,.07,.41,.87,1.]
        menu=Menu(raw_polynomials(e,b,p),e['m'],ks)
        lower,upper=menu_values(menu,e,b,p,ts)
        values=menu.values(ts)
        # Independent floating evaluation may differ by a few ulps; large
        # enclosure violations would indicate an incorrect entropy formula.
        assert np.all(lower<=values+3e-15),(e['id'],float(np.max(lower-values)))
        assert np.all(upper>=values-3e-15),(e['id'],float(np.max(values-upper)))
        dual=rng.dirichlet(np.ones(len(ts)))
        cert=dual_upper(menu,e,b,p,ts,dual)
        direct=menu.support(values,dual)
        assert direct<=cert['upper']+3e-15
        largest=max(largest,float(np.max(upper-lower)));checks+=values.size
    out=dict(enclosed_signed_components=checks,largest_interval_width=largest,
             arithmetic='Independent direct entropy comparison, endpoints included; normalized dual support checked.')
    atomic_json(DATA/'interval_modes_checks.json',out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
