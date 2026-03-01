"""
city_lookup.py
Run once to generate city_lookup.json:
    python city_lookup.py

Reads airline_ticket_dataset.xlsx and computes per-city median values for
the 11 model features, saving the result to city_lookup.json.
"""

import os
import json
import pandas as pd

DATASET_PATH = '/Users/luozihan/Desktop/Airline Tickets/airline_ticket_dataset.xlsx'
OUTPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'city_lookup.json')

# Features to compute from each city's perspective
CITY1_COLS = [
    'nsmiles', 'passengers', 'large_ms', 'lf_ms', 'fare_low',
    'TotalFaredPax_city1', 'TotalPerLFMkts_city1', 'TotalPerPrem_city1',
]
CITY2_COLS = [
    'TotalFaredPax_city2', 'TotalPerLFMkts_city2', 'TotalPerPrem_city2',
]


def build_lookup():
    print(f"Reading dataset from: {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH)
    print(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    # Validate required columns exist
    required = {'city1', 'city2'} | set(CITY1_COLS) | set(CITY2_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    city1_medians = df.groupby('city1')[CITY1_COLS].median()
    city2_medians = df.groupby('city2')[CITY2_COLS].median()

    all_cities = set(df['city1'].dropna().unique()) | set(df['city2'].dropna().unique())
    print(f"Found {len(all_cities):,} unique cities")

    lookup = {}
    for city in all_cities:
        entry = {}
        if city in city1_medians.index:
            entry.update(city1_medians.loc[city].to_dict())
        if city in city2_medians.index:
            entry.update(city2_medians.loc[city].to_dict())
        # Convert numpy types to plain Python floats for JSON serialization
        lookup[city] = {k: float(v) for k, v in entry.items()}

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(lookup, f, indent=2, sort_keys=True)

    print(f"Wrote {len(lookup):,} city entries to: {OUTPUT_PATH}")


if __name__ == '__main__':
    build_lookup()
