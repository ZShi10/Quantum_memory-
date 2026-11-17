"""Extract the author's saved curves without recomputation or interpolation."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'sources' / 'sun'
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    f = np.linspace(.5, 1, 500)  # visualizer_werner.ipynb, code cell 3
    curves = {}
    arrays = []
    for path in sorted((source / 'data' / 'werner').glob('bb84_key_rate_*.npy')):
        a = np.load(path, allow_pickle=False)
        assert a.shape[0] == len(f) and np.isfinite(a).all(), path
        name = path.stem.removeprefix('bb84_key_rate_')
        curves[name] = a.reshape(len(f), -1).max(axis=1)
        arrays.append(dict(name=name, shape=list(a.shape), dtype=str(a.dtype),
                           minimum=float(a.min()), maximum=float(a.max())))
    names = list(curves)
    with (out / 'sun_author_saved_curves.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['F', 'Q'] + names)
        for i in reversed(range(len(f))):
            writer.writerow([f[i], 2*(1-f[i])/3] + [curves[name][i] for name in names])
    record = dict(status='author_arrays_extracted', arrays=arrays, points_per_curve=len(f),
                  missing=['4-2-1 saved rate array'],
                  rate_convention='Author BB84 separate bit/phase entropy bound, evaluated on Werner input; not a generic BB84-compatible-state minimax curve.',
                  fidelity_grid='np.linspace(0.5, 1, 500), directly from author visualizer',
                  operations='Max over saved candidate axes, as in author visualizer cell 8; F to Q exact relation Q=2(1-F)/3; rows reversed for increasing Q.',
                  completed_figure_reproduction=False,
                  prohibition='Do not label the envelope of these eight sizes as the complete nine-size Sun result. The missing 4-2-1 curve requires separate faithful reconstruction.')
    (out / 'sun_data_audit.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
