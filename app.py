"""
app.py
Flask backend for the airfare prediction tool.

Requires:
  - model.pkl in the same directory as this file
  - city_lookup.json in the same directory (generate with city_lookup.py)

Run:
    python app.py

Listens on: http://localhost:5001
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
LOOKUP_PATH = os.path.join(BASE_DIR, 'city_lookup.json')
STATS_PATH  = os.path.join(BASE_DIR, 'model_stats.json')

# Exact feature order required by the model
FEATURE_ORDER = [
    'nsmiles', 'passengers', 'large_ms', 'lf_ms', 'fare_low',
    'TotalFaredPax_city1', 'TotalPerLFMkts_city1', 'TotalPerPrem_city1',
    'TotalFaredPax_city2', 'TotalPerLFMkts_city2', 'TotalPerPrem_city2',
]

# Which city's lookup to use when a feature is missing from the request
CITY1_FEATURES = frozenset([
    'nsmiles', 'passengers', 'large_ms', 'lf_ms', 'fare_low',
    'TotalFaredPax_city1', 'TotalPerLFMkts_city1', 'TotalPerPrem_city1',
])
CITY2_FEATURES = frozenset([
    'TotalFaredPax_city2', 'TotalPerLFMkts_city2', 'TotalPerPrem_city2',
])

# DOT data orders cities alphabetically, so many cities only appear as city1.
# Fall back to their city1 equivalents (same airport stat, different label).
CITY2_FALLBACK = {
    'TotalFaredPax_city2':  'TotalFaredPax_city1',
    'TotalPerLFMkts_city2': 'TotalPerLFMkts_city1',
    'TotalPerPrem_city2':   'TotalPerPrem_city1',
}

# ---------------------------------------------------------------------------
# Load assets once at startup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

with open(MODEL_PATH, 'rb') as _f:
    MODEL = pickle.load(_f)

with open(LOOKUP_PATH, 'r') as _f:
    CITY_LOOKUP = json.load(_f)

print(f"Model loaded from: {MODEL_PATH}")
print(f"City lookup loaded: {len(CITY_LOOKUP):,} cities from {LOOKUP_PATH}")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    city1 = str(body.get('city1', '')).strip()
    city2 = str(body.get('city2', '')).strip()

    if not city1:
        return jsonify({'error': 'city1 is required'}), 400
    if not city2:
        return jsonify({'error': 'city2 is required'}), 400
    if city1 not in CITY_LOOKUP:
        return jsonify({'error': f'city1 "{city1}" not found in lookup'}), 404
    if city2 not in CITY_LOOKUP:
        return jsonify({'error': f'city2 "{city2}" not found in lookup'}), 404

    c1_data = CITY_LOOKUP[city1]
    c2_data = CITY_LOOKUP[city2]

    feature_values   = {}
    missing_features = []

    for feat in FEATURE_ORDER:
        user_val = body.get(feat)
        if user_val is not None:
            try:
                feature_values[feat] = float(user_val)
            except (ValueError, TypeError):
                return jsonify({'error': f'Feature "{feat}" must be numeric, got: {user_val!r}'}), 400
        elif feat in CITY1_FEATURES:
            if feat in c1_data:
                feature_values[feat] = c1_data[feat]
            else:
                missing_features.append(feat)
        else:  # CITY2_FEATURES
            if feat in c2_data:
                feature_values[feat] = c2_data[feat]
            elif CITY2_FALLBACK[feat] in c2_data:
                # City only appears as city1 in the dataset; reuse its city1 stat
                feature_values[feat] = c2_data[CITY2_FALLBACK[feat]]
            else:
                missing_features.append(feat)

    if missing_features:
        return jsonify({
            'error': (
                f'Features {missing_features} could not be resolved. '
                'They are missing from the city lookup and were not provided in the request.'
            )
        }), 422

    feature_array = np.array(
        [[feature_values[f] for f in FEATURE_ORDER]],
        dtype=np.float64,
    )

    predicted_fare     = round(float(MODEL.predict(feature_array)[0]), 2)
    nsmiles            = feature_values['nsmiles']
    baseline_fare      = round(60.0 + 0.055 * nsmiles, 2)
    structural_premium = round(predicted_fare - baseline_fare, 2)

    return jsonify({
        'predicted_fare':      predicted_fare,
        'baseline_fare':       baseline_fare,
        'structural_premium':  structural_premium,
        'feature_values_used': {f: feature_values[f] for f in FEATURE_ORDER},
    })


@app.route('/stats', methods=['GET'])
def stats():
    """Returns model performance metrics from model_stats.json."""
    with open(STATS_PATH, 'r') as f:
        return jsonify(json.load(f))


@app.route('/cities', methods=['GET'])
def list_cities():
    """Returns sorted list of all city names known to the lookup."""
    return jsonify({'cities': sorted(CITY_LOOKUP.keys())})


@app.route('/health', methods=['GET'])
def health():
    """Liveness check."""
    return jsonify({'status': 'ok', 'cities_loaded': len(CITY_LOOKUP)})


@app.route('/')
@app.route('/<path:filename>')
def static_files(filename='airfare_tool.html'):
    return send_from_directory(BASE_DIR, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
