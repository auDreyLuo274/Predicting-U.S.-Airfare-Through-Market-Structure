import os
import json
import pickle

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths (relative to this file) ────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model.pkl')
LOOKUP_PATH = os.path.join(BASE_DIR, 'city_lookup.json')
STATS_PATH  = os.path.join(BASE_DIR, 'model_stats.json')

# ── Feature order must match training ────────────────────────────────────────
FEATURES = [
    'nsmiles', 'passengers', 'large_ms', 'lf_ms', 'fare_low',
    'TotalFaredPax_city1', 'TotalPerLFMkts_city1', 'TotalPerPrem_city1',
    'TotalFaredPax_city2', 'TotalPerLFMkts_city2', 'TotalPerPrem_city2',
]

# Route-level features stored per-city in lookup (averaged for a given pair)
ROUTE_FEATS = ['nsmiles', 'passengers', 'large_ms', 'lf_ms', 'fare_low']

# Hub features stored with _city1 suffix in lookup; remapped for city2
HUB_FEATS = ['TotalFaredPax', 'TotalPerLFMkts', 'TotalPerPrem']

# ── Load artifacts ────────────────────────────────────────────────────────────
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(LOOKUP_PATH, 'r') as f:
    city_lookup = json.load(f)

with open(STATS_PATH, 'r') as f:
    model_stats = json.load(f)

# Fallback medians for any missing values
MEDIANS = {feat: model_stats['feature_stats'][feat]['median'] for feat in FEATURES}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)


@app.route('/cities', methods=['GET'])
def get_cities():
    """Return sorted list of all city names."""
    return jsonify({'cities': sorted(city_lookup.keys())})


@app.route('/stats', methods=['GET'])
def get_stats():
    """Return model stats (SHAP values + feature stats) for the frontend."""
    return jsonify(model_stats)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept {city1, city2}, build feature vector, run model.
    Returns {predicted_fare, baseline_fare, structural_premium, feature_values_used}.
    """
    data = request.get_json(force=True)
    city1 = data.get('city1', '').strip()
    city2 = data.get('city2', '').strip()

    if not city1 or not city2:
        return jsonify({'error': 'city1 and city2 are required'}), 400
    if city1 not in city_lookup:
        return jsonify({'error': f'City not found: {city1}'}), 404
    if city2 not in city_lookup:
        return jsonify({'error': f'City not found: {city2}'}), 404

    c1 = city_lookup[city1]
    c2 = city_lookup[city2]

    # Build feature vector
    fv = {}

    # Route-level features: average the two cities' medians
    for feat in ROUTE_FEATS:
        v1 = c1.get(feat, MEDIANS[feat])
        v2 = c2.get(feat, MEDIANS[feat])
        fv[feat] = (v1 + v2) / 2.0

    # Hub features for city1 (stored as TotalFaredPax_city1, etc.)
    for hub in HUB_FEATS:
        key = f'{hub}_city1'
        fv[key] = c1.get(key, MEDIANS[key])

    # Hub features for city2 (lookup stores them as _city1; remap to _city2)
    for hub in HUB_FEATS:
        lookup_key = f'{hub}_city1'   # key name used in city_lookup
        feat_key   = f'{hub}_city2'   # key name expected by model
        fv[feat_key] = c2.get(lookup_key, MEDIANS[feat_key])

    # Build numpy array in exact feature order
    X = np.array([[fv[feat] for feat in FEATURES]], dtype=float)

    predicted_fare     = float(model.predict(X)[0])
    baseline_fare      = 60.0 + 0.055 * fv['nsmiles']
    structural_premium = predicted_fare - baseline_fare

    # Round feature values for readability
    feature_values_used = {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in fv.items()
    }

    return jsonify({
        'predicted_fare':      round(predicted_fare, 2),
        'baseline_fare':       round(baseline_fare, 2),
        'structural_premium':  round(structural_premium, 2),
        'feature_values_used': feature_values_used,
    })


@app.route('/', defaults={'path': 'airfare_tool.html'})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(BASE_DIR, path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
