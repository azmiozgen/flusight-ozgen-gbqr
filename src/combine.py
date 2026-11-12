"""
Score quantile averages (Vincentization) of saved backtest forecasts.

Usage: python -I combine.py HUB_DIR BT_DIR MODEL_A MODEL_B [WEIGHTS...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import Hub  # noqa: E402
from wis import relative_wis, score_table  # noqa: E402

KEY = ['reference_date', 'location', 'horizon', 'target_end_date', 'q']


def load(bt, m):
    f = pd.read_csv(Path(bt) / f'{m}_forecasts.csv.gz', dtype={'location': str},
                    parse_dates=['reference_date', 'target_end_date'])
    return f[KEY + ['value']]


def widen_quantiles(f, c):
    """Scale each quantile's distance from the median by c, on the log1p scale."""
    key = ['reference_date', 'location', 'horizon']
    u = np.log1p(f['value'].clip(lower=0))
    med = u.where(f['q'] == 0.5).groupby([f[k] for k in key]).transform('max')
    return np.expm1(med + c * (u - med)).clip(lower=0)


def main(hub_dir, bt, a, b, *weights):
    hub = Hub(hub_dir)
    fa, fb = load(bt, a), load(bt, b)
    m = fa.merge(fb, on=KEY, suffixes=('_a', '_b'))
    m = m[m['horizon'] >= 0]
    refs = m['reference_date'].unique()
    ens = hub.model_forecasts('FluSight-ensemble')
    base = hub.model_forecasts('FluSight-baseline')
    truth = hub.truth()
    rows = []
    for name, f in (('FluSight-ensemble', ens), ('FluSight-baseline', base)):
        s = score_table(f[f['reference_date'].isin(refs) & (f['horizon'] >= 0)], truth)
        s['model'] = name
        rows.append(s)
    widen = [float(x) for x in os.environ.get('WIDEN', '1').split(',')]
    for w in [float(x) for x in weights] or [0.5]:
        for c in widen:
            f = m[KEY].copy()
            f['value'] = w * m['value_a'] + (1 - w) * m['value_b']
            f['value'] = np.round(widen_quantiles(f, c))
            s = score_table(f, truth)
            s['model'] = f'w{w:.2f}_c{c:.2f}'
            rows.append(s)
    sc = pd.concat(rows, ignore_index=True)
    sc['season'] = np.where(sc['reference_date'].dt.month >= 8, sc['reference_date'].dt.year,
                            sc['reference_date'].dt.year - 1)
    res = {'all': relative_wis(sc)['rel_wis']}
    for season, d in sc.groupby('season'):
        res[season] = relative_wis(d)['rel_wis']
    cov = sc.groupby('model')[['cov50', 'cov95']].mean()
    print(f'A = {a}, B = {b}; weight is on A')
    print(pd.DataFrame(res).join(cov).round(3).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:])
