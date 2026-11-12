"""
Data access for the FluSight 2026-27 hub: NHSN weekly influenza admission vintages and
locations, read from a clone of cdcepi/FluSight-forecast-hub (data only, no hub code is run).
"""
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
TARGET = 'wk inc flu hosp'


class Hub:
    def __init__(self, hub_dir, extra_vintage=None):
        self.dir = Path(hub_dir)
        loc = pd.read_csv(self.dir / 'auxiliary-data' / 'locations.csv', dtype={'location': str})
        self.locations = loc.set_index('location')
        ts = pd.read_csv(self.dir / 'target-data' / 'time-series.csv',
                         dtype={'location': str}, parse_dates=['as_of', 'target_end_date'])
        if extra_vintage:
            ex = pd.read_csv(extra_vintage, dtype={'location': str},
                             parse_dates=['as_of', 'target_end_date'])
            ts = pd.concat([ts[~ts['as_of'].isin(ex['as_of'].unique())], ex])
        self.vintages = ts[ts['target'] == TARGET][
            ['as_of', 'target_end_date', 'location', 'observation']]
        self.as_of_dates = np.sort(self.vintages['as_of'].unique())

    def snapshot(self, as_of):
        """Weeks x locations table of counts as published on the given vintage."""
        v = self.vintages[self.vintages['as_of'] == pd.Timestamp(as_of)].drop_duplicates()
        return v.pivot(index='target_end_date', columns='location', values='observation')

    def vintage_for(self, reference_date):
        """Latest vintage dated on or before the Wednesday deadline (reference date - 3 days).
        Vintages of 2023-25 are dated by Saturdays and include the week ending that day, so a
        vintage dated on the reference date was not available at the deadline."""
        cut = np.datetime64(pd.Timestamp(reference_date) - pd.Timedelta(days=3))
        ok = self.as_of_dates[self.as_of_dates <= cut]
        return pd.Timestamp(ok[-1]) if len(ok) else None

    def realtime_snapshot(self, reference_date, as_of=None):
        """Data as a forecaster had it at the deadline: the vintage above (or as_of), cut after
        the week ending one week before the reference date."""
        snap = self.snapshot(as_of or self.vintage_for(reference_date))
        return snap[snap.index <= pd.Timestamp(reference_date) - pd.Timedelta(days=7)]

    def truth(self):
        """Latest vintage, used as the evaluation target."""
        return self.snapshot(self.as_of_dates[-1])

    def population(self):
        return self.locations['population']

    def model_forecasts(self, model):
        """All quantile forecasts of `wk inc flu hosp` from one hub model."""
        rows = []
        for f in sorted((self.dir / 'model-output' / model).glob('*.csv')):
            d = pd.read_csv(f, dtype={'location': str, 'output_type_id': str})
            d = d[(d['target'] == TARGET) & (d['output_type'] == 'quantile')]
            rows.append(d)
        out = pd.concat(rows, ignore_index=True)
        out['reference_date'] = pd.to_datetime(out['reference_date'])
        out['target_end_date'] = pd.to_datetime(out['target_end_date'])
        out['q'] = out['output_type_id'].astype(float)
        return out[['reference_date', 'location', 'horizon', 'target_end_date', 'q', 'value']]

    def backfill_factors(self, reference_date, n_pairs=12, min_gap_weeks=4, as_of=None):
        """Median ratio (latest available value / value as first seen at a deadline) for the
        last and second-last reported weeks, per location, from up to n_pairs past rounds whose
        weeks are at least min_gap_weeks old at this deadline. Uses only data available at the
        deadline of reference_date."""
        ref = pd.Timestamp(reference_date)
        now = self.snapshot(as_of or self.vintage_for(ref))
        r1, r2 = [], []
        for j in range(min_gap_weeks, min_gap_weeks + 3 * n_pairs):
            past = ref - pd.Timedelta(days=7 * j)
            if self.vintage_for(past) is None:
                break
            snap = self.realtime_snapshot(past)
            if snap.empty or (past - snap.index.max()).days != 7:
                continue
            w1, w2 = snap.index[-1], snap.index[-2]
            if w1 not in now.index or w2 not in now.index:
                continue
            r1.append(np.log((now.loc[w1] + 1) / (snap.loc[w1] + 1)))
            r2.append(np.log((now.loc[w2] + 1) / (snap.loc[w2] + 1)))
            if len(r1) >= n_pairs:
                break
        if not r1:
            return None
        f1 = np.exp(pd.concat(r1, axis=1).median(axis=1).clip(-0.2, 0.4))
        f2 = np.exp(pd.concat(r2, axis=1).median(axis=1).clip(-0.2, 0.4))
        return pd.DataFrame({'f1': f1, 'f2': f2})

    def adjusted_snapshot(self, reference_date, as_of=None):
        """Real-time snapshot with the last two weeks scaled by the backfill factors."""
        snap = self.realtime_snapshot(reference_date, as_of).copy()
        f = self.backfill_factors(reference_date, as_of=as_of)
        if f is None:
            return snap
        cols = [c for c in snap.columns if c in f.index]
        snap.loc[snap.index[-1], cols] *= f.loc[cols, 'f1'].to_numpy()
        snap.loc[snap.index[-2], cols] *= f.loc[cols, 'f2'].to_numpy()
        return snap
