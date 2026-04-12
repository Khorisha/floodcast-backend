import pandas as pd
import numpy as np
import json
import os

def load_zone_risks():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'zone_risk_lookup.csv')
    df = pd.read_csv(file_path)

    # Detect columns flexibly
    zone_col = None
    risk_col = None
    for col in df.columns:
        cl = col.lower()
        if 'zone' in cl or 'name' in cl:
            zone_col = col
        if 'risk' in cl or 'mean' in cl:
            risk_col = col
    if zone_col is None:
        zone_col = df.columns[0]
    if risk_col is None:
        risk_col = df.columns[1]

    raw = dict(zip(df[zone_col], df[risk_col]))

    # Normalise names to match GeoJSON (e.g. "Port Louis CBD" → "CBD")
    NAME_MAP = {'Port Louis CBD': 'CBD'}
    return {NAME_MAP.get(k, k): v for k, v in raw.items()}

def apply_gis_multiplier(city_prob, zone_risks):
    """
    GIS Post-Hoc Fusion:
        zone_prob_i = baseline_prob × (1 + risk_mean_i / 5)
    Capped at 0.99 to keep it a valid probability.
    """
    zone_probs = {}
    for zone, risk in zone_risks.items():
        zone_probs[zone] = min(0.99, city_prob * (1 + risk / 5))
    return zone_probs

def get_zone_geojson():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'gis_zones.geojson')
    with open(file_path, 'r') as f:
        return json.load(f)

def get_alert_level(prob, wet_season=True):
    if wet_season:
        if prob >= 0.05:
            return 'red', 'Warning - Take action now'
        elif prob >= 0.02:
            return 'orange', 'Advisory - Stay aware'
        elif prob >= 0.01:
            return 'yellow', 'Watch - Be prepared'
        else:
            return 'green', 'Low risk'
    else:
        if prob >= 0.08:
            return 'red', 'Warning - Take action now'
        elif prob >= 0.03:
            return 'orange', 'Advisory - Stay aware'
        elif prob >= 0.01:
            return 'yellow', 'Watch - Be prepared'
        else:
            return 'green', 'Low risk'