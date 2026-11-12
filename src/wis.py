"""
Weighted interval score from the 23 FluSight quantiles (Bracher et al. 2021), computed as the
mean quantile (pinball) loss times 2, which equals WIS for symmetric quantile sets including
the median; and central-interval coverage.
"""
import numpy as np
import pandas as pd

from data import QUANTILES


def wis(q_values, y):
    """q_values: (n, 23) quantiles in QUANTILES order; y: (n,) observations."""
    q_values, y = np.asarray(q_values, float), np.asarray(y, float)[:, None]
    tau = QUANTILES[None, :]
    loss = np.where(y >= q_values, tau * (y - q_values), (1 - tau) * (q_values - y))
    return 2 * loss.mean(axis=1)


def score_table(fc, truth):
    """fc: long table (reference_date, location, horizon, target_end_date, q, value).
    truth: weeks x locations. Returns one row per forecast with WIS, AE and coverage."""
    wide = fc.pivot_table(index=['reference_date', 'location', 'horizon', 'target_end_date'],
                          columns='q', values='value').reset_index()
    qcols = [c for c in wide.columns if isinstance(c, float)]
    if len(qcols) != len(QUANTILES):
        raise ValueError('missing quantiles')
    t = truth.stack().rename('y').reset_index()
    t.columns = ['target_end_date', 'location', 'y']
    wide = wide.merge(t, on=['target_end_date', 'location'], how='inner')
    qv = wide[sorted(qcols)].to_numpy()
    wide['wis'] = wis(qv, wide['y'])
    wide['ae'] = np.abs(qv[:, 11] - wide['y'])
    wide['cov50'] = (qv[:, 6] <= wide['y']) & (wide['y'] <= qv[:, 16])
    wide['cov95'] = (qv[:, 1] <= wide['y']) & (wide['y'] <= qv[:, 21])
    return wide[['reference_date', 'location', 'horizon', 'target_end_date', 'y', 'wis', 'ae',
                 'cov50', 'cov95']]


def relative_wis(scores, model_col='model', base='FluSight-baseline'):
    """Pairwise-free relative WIS on the common forecast set: mean WIS / mean baseline WIS."""
    key = ['reference_date', 'location', 'horizon']
    piv = scores.pivot_table(index=key, columns=model_col, values='wis').dropna()
    out = pd.DataFrame({'n': len(piv), 'mean_wis': piv.mean()})
    out['rel_wis'] = out['mean_wis'] / out.loc[base, 'mean_wis']
    return out.sort_values('rel_wis')
