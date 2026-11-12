"""
Fallback when the hub's target data is not yet updated: fetch the preliminary NHSN dataset
(data.cdc.gov mpgq-jmmr, the source of the hub's target data) and write it as one vintage in
the layout of target-data/time-series.csv, following target-data/get_target_data.R: weeks from
2022-02-01, territories VI/AS/GU/MP and regions dropped, USA -> US, abbreviations -> FIPS.

Usage: python -I fetch_nhsn.py HUB_DIR AS_OF OUT_CSV
"""
import json
import sys
import urllib.request

import pandas as pd

URL = ('https://data.cdc.gov/resource/mpgq-jmmr.json?$select=weekendingdate,jurisdiction,'
       'totalconfflunewadm&$where=weekendingdate>=%272022-02-01%27&$limit=50000')


def main(hub_dir, as_of, out_csv):
    with urllib.request.urlopen(URL, timeout=120) as r:
        rows = json.loads(r.read())
    d = pd.DataFrame(rows)
    d = d[~d['jurisdiction'].isin(['VI', 'AS', 'GU', 'MP'])
          & ~d['jurisdiction'].str.contains('Region')]
    d['jurisdiction'] = d['jurisdiction'].replace({'USA': 'US'})
    loc = pd.read_csv(f'{hub_dir}/auxiliary-data/locations.csv', dtype={'location': str})
    d = d.merge(loc[['abbreviation', 'location', 'location_name', 'population']],
                left_on='jurisdiction', right_on='abbreviation')
    d['observation'] = pd.to_numeric(d['totalconfflunewadm'])
    out = pd.DataFrame({
        'as_of': as_of, 'target': 'wk inc flu hosp',
        'target_end_date': pd.to_datetime(d['weekendingdate']).dt.strftime('%Y-%m-%d'),
        'location': d['location'], 'location_name': d['location_name'],
        'observation': d['observation'],
        'weekly_rate': d['observation'] * 1e5 / d['population']})
    out = out.sort_values(['target_end_date', 'location'])
    out.to_csv(out_csv, index=False)
    print(len(out), 'rows; last week', out['target_end_date'].max(), '; locations',
          out['location'].nunique())


if __name__ == '__main__':
    main(*sys.argv[1:4])
