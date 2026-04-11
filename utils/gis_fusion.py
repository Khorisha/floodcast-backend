import pandas as pd
import numpy as np
import json
import os

def load_zone_risks():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'zone_risk_lookup.csv')
    df = pd.read_csv(file_path)
    
    # Print columns to debug
    print("CSV columns:", df.columns.tolist())
    
    # Try different possible column names
    zone_col = None
    risk_col = None
    
    for col in df.columns:
        if 'zone' in col.lower() or 'Zone' in col or 'name' in col.lower():
            zone_col = col
        if 'risk' in col.lower() or 'Risk' in col or 'mean' in col.lower():
            risk_col = col
    
    if zone_col is None:
        zone_col = df.columns[0]  # First column as zone name
    if risk_col is None:
        risk_col = df.columns[1]  # Second column as risk
    
    print(f"Using zone column: {zone_col}, risk column: {risk_col}")
    
    return dict(zip(df[zone_col], df[risk_col]))

def apply_gis_multiplier(city_prob, zone_risks, base_risk=3.87):
    zone_probs = {}
    for zone, risk in zone_risks.items():
        multiplier = 1 + (risk - base_risk) / base_risk
        multiplier = max(0.7, min(1.5, multiplier))
        zone_probs[zone] = min(0.99, city_prob * multiplier)
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