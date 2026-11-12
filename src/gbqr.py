"""
Global gradient-boosted quantile regression for weekly influenza admissions (in the spirit of
UMass-flusion's GBQR component, re-implemented from its description, not its code).

Scale: z = (admissions per 100k) ** (1/4). For the last observed week t of a vintage and a step
s = 1..4 weeks ahead, the target is z[t+s] - z[t]. Features describe the recent trajectory of
the location; one model per quantile level is trained on all locations and steps, using only
the vintage available at the reference date. Horizon -1 (the last reported week, not scored) is
the reported value with a narrow band.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from data import QUANTILES

POWER = 0.25
STEPS = (1, 2, 3, 4)


def to_z(counts, pop):
    return (np.clip(counts, 0, None) / pop * 1e5) ** POWER


def from_z(z, pop):
    return np.clip(z, 0, None) ** (1 / POWER) * pop / 1e5


def features(z):
    """z: weeks x locations (transformed). Returns long feature table for every (week, loc)."""
    f = {
        'z0': z,
        'd1': z - z.shift(1),
        'd2': z - z.shift(2),
        'd4': z - z.shift(4),
        'm4': z.rolling(4, min_periods=3).mean(),
        'sd4': z.diff().rolling(4, min_periods=3).std(),
        'max52': z.rolling(52, min_periods=20).max(),
    }
    # slope of a linear fit over the last 4 weeks
    w = np.array([-1.5, -0.5, 0.5, 1.5]) / 5.0
    f['slope4'] = sum(w[i] * z.shift(3 - i) for i in range(4))
    long = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    long.index.names = ['date', 'location']
    long = long.reset_index()
    wk = long['date'].dt.isocalendar().week.astype(int)
    long['season_week'] = (wk - 40) % 52
    return long


FEATURES = ['z0', 'd1', 'd2', 'd4', 'm4', 'sd4', 'max52', 'slope4', 'season_week', 'step',
            'logpop']


def training_table(z):
    feat = features(z)
    rows = []
    for s in STEPS:
        tgt = (z.shift(-s) - z).stack(future_stack=True).rename('y').reset_index()
        tgt.columns = ['date', 'location', 'y']
        d = feat.merge(tgt, on=['date', 'location'])
        d['step'] = s
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return d.dropna(subset=['z0', 'd1', 'y'])


def fit_predict(snapshot, pop, reference_date, max_iter=250, seed=0, drop=()):
    """Quantile forecasts (long table) for horizons -1..3 from one vintage."""
    snapshot = snapshot.sort_index()
    locs = [c for c in snapshot.columns if c in pop.index]
    snapshot = snapshot[locs]
    z = to_z(snapshot, pop[locs])
    last = snapshot.index.max()
    train = training_table(z)
    train['logpop'] = np.log10(pop.loc[train['location']].to_numpy())
    train = train[train['date'] + pd.to_timedelta(7 * train['step'], 'D') <= last]
    feat = features(z)
    cur = feat[feat['date'] == last].copy()
    cur['logpop'] = np.log10(pop.loc[cur['location']].to_numpy())
    ref = pd.Timestamp(reference_date)
    out = []
    preds = {}
    test_rows = []
    for h in (0, 1, 2, 3):
        ted = ref + pd.Timedelta(days=7 * h)
        s = int((ted - last).days // 7)
        c = cur.copy()
        c['step'] = s
        c['horizon'] = h
        c['target_end_date'] = ted
        test_rows.append(c)
    test = pd.concat(test_rows, ignore_index=True)
    cols = [c for c in FEATURES if c not in drop]
    xtr, ytr = train[cols].to_numpy(float), train['y'].to_numpy(float)
    xte = test[cols].to_numpy(float)
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=max_iter,
                                          learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, random_state=seed)
        m.fit(xtr, ytr)
        preds[q] = m.predict(xte)
    delta = np.sort(np.column_stack([preds[q] for q in QUANTILES]), axis=1)
    pops = pop.loc[test['location']].to_numpy()[:, None]
    vals = from_z(test['z0'].to_numpy()[:, None] + delta, pops)
    for i, q in enumerate(QUANTILES):
        d = test[['location', 'horizon', 'target_end_date']].copy()
        d['q'] = q
        d['value'] = vals[:, i]
        out.append(d)
    # horizon -1: last reported week, +-5% band (not scored)
    prev = ref - pd.Timedelta(days=7)
    if prev in snapshot.index:
        lastrow = snapshot.loc[prev]
        for q in QUANTILES:
            f = 1 + 0.1 * (q - 0.5)
            out.append(pd.DataFrame({'location': lastrow.index, 'horizon': -1,
                                     'target_end_date': prev, 'q': q,
                                     'value': lastrow.to_numpy(float) * f}))
    fc = pd.concat(out, ignore_index=True)
    fc['reference_date'] = ref
    fc = fc.dropna(subset=['value'])
    return fc[['reference_date', 'location', 'horizon', 'target_end_date', 'q', 'value']]
