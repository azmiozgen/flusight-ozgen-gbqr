"""
Local checks of a FluSight submission file, mirroring the hub rules for `wk inc flu hosp`
quantiles (hub-config/tasks.json, model-output README): file name, columns, task ids, the 23
required quantiles per task, integer non-negative values that do not decrease with the
quantile level, target_end_date = reference_date + 7 * horizon, the plausibility bound (at most
30% of the population), and the submission window (reference_date - 6 to - 2 days).

Usage: python -I validate.py HUB_DIR SUBMISSION_CSV [TODAY]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['reference_date', 'target', 'horizon', 'target_end_date', 'location', 'output_type',
           'output_type_id', 'value']


def main(hub_dir, path, today=None):
    hub, path = Path(hub_dir), Path(path)
    errors = []
    m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})-([A-Za-z0-9_]+)-([A-Za-z0-9_]+)\.csv', path.name)
    if not m:
        errors.append(f'bad file name {path.name}')
    ref = pd.Timestamp(m.group(1)) if m else None
    if path.parent.name != f'{m.group(2)}-{m.group(3)}' if m else True:
        errors.append(f'folder {path.parent.name} does not match the file name')
    tasks = json.loads((hub / 'hub-config' / 'tasks.json').read_text())
    mt = [t for t in tasks['rounds'][0]['model_tasks']
          if 'wk inc flu hosp' in t['task_ids']['target'].get('optional', [])][0]
    ids = mt['task_ids']
    req_q = mt['output_type']['quantile']['output_type_id']['required']
    d = pd.read_csv(path, dtype={'location': str, 'output_type_id': str})
    if list(d.columns) != COLUMNS and set(d.columns) != set(COLUMNS):
        errors.append(f'columns {list(d.columns)}')
    if set(d['target']) != {'wk inc flu hosp'} or set(d['output_type']) != {'quantile'}:
        errors.append('unexpected target or output type')
    if set(d['reference_date']) != {m.group(1)}:
        errors.append('reference_date differs from the file name')
    for col in ('horizon', 'location', 'reference_date', 'target_end_date'):
        allowed = {str(x) for x in ids[col].get('optional', []) + (ids[col]['required'] or [])}
        bad = set(d[col].astype(str)) - allowed
        if bad:
            errors.append(f'{col} values not allowed: {sorted(map(str, bad))[:5]}')
    ted = pd.to_datetime(d['reference_date']) + pd.to_timedelta(7 * d['horizon'], 'D')
    if (ted.dt.strftime('%Y-%m-%d') != d['target_end_date']).any():
        errors.append('target_end_date != reference_date + 7 * horizon')
    d['q'] = d['output_type_id'].astype(float)
    if set(np.round(d['q'], 6)) != set(np.round(req_q, 6)):
        errors.append('quantile levels differ from the required 23')
    g = d.groupby(['location', 'horizon'])
    if (g.size() != len(req_q)).any():
        errors.append('a task does not have exactly 23 quantiles')
    if d.duplicated(['location', 'horizon', 'output_type_id']).any():
        errors.append('duplicate rows')
    v = d['value']
    if (v < 0).any() or (v != np.round(v)).any() or v.isna().any():
        errors.append('values must be non-negative integers')
    dec = d.sort_values('q').groupby(['location', 'horizon'])['value'].apply(
        lambda s: (np.diff(s.to_numpy()) < 0).any())
    if dec.any():
        errors.append(f'quantiles decrease for {dec[dec].index.tolist()[:5]}')
    pop = pd.read_csv(hub / 'auxiliary-data' / 'locations.csv', dtype={'location': str})
    pop = pop.set_index('location')['population']
    over = d[d['value'] > 0.3 * pop.loc[d['location']].to_numpy()]
    if len(over):
        errors.append(f'{len(over)} values above 30% of the population')
    if today is not None:
        t = pd.Timestamp(today)
        if not (ref - pd.Timedelta(days=6) <= t <= ref - pd.Timedelta(days=2)):
            errors.append(f'{t.date()} outside the submission window for {ref.date()}')
    n_loc, n_h = d['location'].nunique(), d['horizon'].nunique()
    print(f'{path.name}: {len(d)} rows, {n_loc} locations, horizons {sorted(d.horizon.unique())}')
    for e in errors:
        print('ERROR', e)
    print('OK' if not errors else f'{len(errors)} errors')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:4]))
