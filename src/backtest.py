"""
Backtest a model on past FluSight rounds, using the data vintage available at each reference
date, and compare WIS with FluSight-ensemble and FluSight-baseline on the common forecasts.

Usage: python -I backtest.py HUB_DIR OUT_DIR [MODEL] [FIRST_REF] [JOBS]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gbqr  # noqa: E402
import trend  # noqa: E402
from data import Hub  # noqa: E402
from wis import relative_wis, score_table  # noqa: E402

MODELS = {
    'gbqr': lambda s, p, r: gbqr.fit_predict(s, p, r, drop=('logpop',)),
    'gbqr_noseason': lambda s, p, r: gbqr.fit_predict(s, p, r, drop=('season_week', 'logpop')),
    'gbqr_nosm': lambda s, p, r: gbqr.fit_predict(s, p, r, drop=('season_week', 'max52', 'logpop')),
    'trend': trend.fit_predict,
    'gbqr_pop': lambda s, p, r: gbqr.fit_predict(s, p, r, drop=('season_week', 'max52')),
    'gbqr_full': lambda s, p, r: gbqr.fit_predict(s, p, r),
    'trend_phi1': lambda s, p, r: trend.fit_predict(s, p, r, phi=1.0),
    'trend_phi6': lambda s, p, r: trend.fit_predict(s, p, r, phi=0.6),
    'trend_k3': lambda s, p, r: trend.fit_predict(s, p, r, k=3),
}


def one_round(hub, model, ref):
    as_of = hub.vintage_for(ref)
    if model.endswith('_bf'):
        snap, model = hub.adjusted_snapshot(ref), model[:-3]
    else:
        snap = hub.realtime_snapshot(ref)
    lag = (ref - snap.index.max()).days // 7
    fc = MODELS[model](snap, hub.population(), ref)
    fc['model'] = model
    fc['value'] = np.round(fc['value'])
    return fc, {'reference_date': ref, 'as_of': as_of, 'lag_weeks': lag}


def main(hub_dir, out_dir, model='gbqr', first_ref='2023-10-14', jobs=10):
    hub = Hub(hub_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ens = hub.model_forecasts('FluSight-ensemble')
    base = hub.model_forecasts('FluSight-baseline')
    refs = sorted(r for r in ens['reference_date'].unique() if r >= pd.Timestamp(first_ref))
    res = Parallel(n_jobs=int(jobs))(delayed(one_round)(hub, model, pd.Timestamp(r))
                                     for r in refs)
    fc = pd.concat([r[0] for r in res], ignore_index=True)
    pd.DataFrame([r[1] for r in res]).to_csv(out / f'{model}_vintages.csv', index=False)
    fc.to_csv(out / f'{model}_forecasts.csv.gz', index=False)
    truth = hub.truth()
    scores = []
    for name, f in ((model, fc), ('FluSight-ensemble', ens), ('FluSight-baseline', base)):
        f = f[f['reference_date'].isin(refs) & (f['horizon'] >= 0)]
        s = score_table(f, truth)
        s['model'] = name
        scores.append(s)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(out / f'{model}_scores.csv.gz', index=False)
    season = np.where(scores['reference_date'].dt.month >= 8, scores['reference_date'].dt.year,
                      scores['reference_date'].dt.year - 1)
    scores['season'] = season
    lines = []
    for name, d in [('all', scores)] + list(scores.groupby('season')):
        r = relative_wis(d)
        cov = d.groupby('model')[['cov50', 'cov95']].mean()
        lines.append(f'== {name}\n{r.join(cov).round(3).to_string()}\n')
    for h, d in scores.groupby('horizon'):
        lines.append(f'== horizon {h}\n{relative_wis(d).round(3).to_string()}\n')
    text = '\n'.join(lines)
    (out / f'{model}_summary.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    main(*sys.argv[1:6])
