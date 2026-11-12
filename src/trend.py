"""
Damped log-linear trend model with empirical error quantiles.

On u = log(admissions + 1) of each location, the slope b is a least-squares fit over the last
K weeks. The point forecast s weeks ahead is u[t] + b * (phi + ... + phi**s). Quantiles add the
empirical quantiles of the same method's past errors on the vintage (all locations and weeks
before the last s weeks), per step and per level bin of u[t] (error spread shrinks with size).
"""
import numpy as np
import pandas as pd

from data import QUANTILES

STEPS = (1, 2, 3, 4)
LEVEL_BINS = np.array([np.log(11), np.log(51), np.log(201)])


def slope(u, k):
    x = np.arange(k) - (k - 1) / 2
    x = x / (x ** 2).sum()
    return sum(x[i] * u.shift(k - 1 - i) for i in range(k))


def damp_sum(phi, s):
    return sum(phi ** i for i in range(1, s + 1))


def fit_predict(snapshot, pop, reference_date, k=4, phi=0.8, min_errors=200):
    snapshot = snapshot.sort_index()
    locs = [c for c in snapshot.columns if c in pop.index]
    u = np.log1p(snapshot[locs].clip(lower=0))
    b = slope(u, k)
    last = snapshot.index.max()
    ref = pd.Timestamp(reference_date)
    level = np.digitize(u.loc[last].to_numpy(), LEVEL_BINS)
    out = []
    for h in (0, 1, 2, 3):
        ted = ref + pd.Timedelta(days=7 * h)
        s = int((ted - last).days // 7)
        mu = u + b * damp_sum(phi, s)
        err = (u.shift(-s) - mu)
        lev = pd.DataFrame(np.digitize(u.to_numpy(), LEVEL_BINS), index=u.index, columns=u.columns)
        e = pd.DataFrame({'err': err.stack(), 'lev': lev.stack()}).dropna()
        qs = {}
        for lv in range(len(LEVEL_BINS) + 1):
            ee = e.loc[e['lev'] == lv, 'err']
            if len(ee) < min_errors:
                ee = e['err']
            qs[lv] = np.quantile(ee.to_numpy(), QUANTILES)
        cur = mu.loc[last].to_numpy()
        vals = np.array([cur[i] + qs[level[i]] for i in range(len(locs))])
        vals = np.expm1(np.maximum.accumulate(vals, axis=1)).clip(min=0)
        for j, q in enumerate(QUANTILES):
            out.append(pd.DataFrame({'location': locs, 'horizon': h, 'target_end_date': ted,
                                     'q': q, 'value': vals[:, j]}))
    fc = pd.concat(out, ignore_index=True)
    fc['reference_date'] = ref
    fc = fc.dropna(subset=['value'])
    return fc[['reference_date', 'location', 'horizon', 'target_end_date', 'q', 'value']]
