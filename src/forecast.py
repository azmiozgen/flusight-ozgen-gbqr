"""
Weekly FluSight submission of Ozgen-GBQR: quantile forecasts of `wk inc flu hosp` for
horizons -1..3 and all hub locations.

Model: 0.7 * GBQR(all features) + 0.3 * GBQR(without season week and 52-week maximum), averaged
by quantile; inputs are the latest vintage with the last two weeks scaled by backfill factors;
each quantile's distance from the median is then widened by 1.3 on the log1p scale.

Usage: python -I forecast.py HUB_DIR REFERENCE_DATE OUT_CSV [AS_OF] [EXTRA_VINTAGE_CSV]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gbqr  # noqa: E402
from combine import widen_quantiles  # noqa: E402
from data import QUANTILES, TARGET, Hub  # noqa: E402

WEIGHT = 0.7
WIDEN = 1.3
KEY = ['reference_date', 'location', 'horizon', 'target_end_date', 'q']


def forecast(hub, ref, as_of):
    snap = hub.adjusted_snapshot(ref, as_of)
    pop = hub.population()
    a = gbqr.fit_predict(snap, pop, ref)
    b = gbqr.fit_predict(snap, pop, ref, drop=('season_week', 'max52'))
    m = a.merge(b, on=KEY, suffixes=('_a', '_b'))
    f = m[KEY].copy()
    f['value'] = WEIGHT * m['value_a'] + (1 - WEIGHT) * m['value_b']
    f['value'] = widen_quantiles(f, WIDEN)
    f = f.sort_values(KEY)
    # integers, non-decreasing in q within each task
    f['value'] = np.round(f['value']).astype(int)
    f['value'] = f.groupby(['location', 'horizon'])['value'].cummax()
    return f


def to_hub_format(f):
    out = pd.DataFrame({
        'reference_date': f['reference_date'].dt.strftime('%Y-%m-%d'),
        'target': TARGET,
        'horizon': f['horizon'].astype(int),
        'target_end_date': f['target_end_date'].dt.strftime('%Y-%m-%d'),
        'location': f['location'],
        'output_type': 'quantile',
        'output_type_id': f['q'].map(lambda q: f'{q:g}'),
        'value': f['value'].astype(int),
    })
    return out.sort_values(['location', 'horizon', 'output_type_id']).reset_index(drop=True)


def main(hub_dir, reference_date, out_csv, as_of=None, extra=None):
    hub = Hub(hub_dir, extra)
    ref = pd.Timestamp(reference_date)
    as_of = pd.Timestamp(as_of) if as_of else pd.Timestamp(hub.as_of_dates[-1])
    snap = hub.realtime_snapshot(ref, as_of)
    last = snap.index.max()
    print('vintage', as_of.date(), 'last week', last.date(), 'locations', snap.shape[1])
    if last != ref - pd.Timedelta(days=7):
        print('WARNING: last reported week is not reference_date - 7 days')
    f = forecast(hub, ref, as_of)
    out = to_hub_format(f)
    assert len(QUANTILES) == out['output_type_id'].nunique()
    out.to_csv(out_csv, index=False)
    print(out.shape, 'rows written to', out_csv)


if __name__ == '__main__':
    main(*sys.argv[1:6])
