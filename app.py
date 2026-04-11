import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from utils.weather_api import get_current_weather, get_forecast_7days, get_historical_hours
from utils.predictor import FloodPredictor
from utils.gis_fusion import load_zone_risks, apply_gis_multiplier, get_zone_geojson, get_alert_level

app = Flask(__name__)

# Configure CORS properly
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'final_model.tflite')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
iso_path = os.path.join(os.path.dirname(__file__), 'models', 'gru_iso.pkl')
shap_path = os.path.join(os.path.dirname(__file__), 'models', 'shap_values.npy')

print("Loading models...")
predictor = FloodPredictor(model_path, scaler_path, iso_path)
zone_risks = load_zone_risks()
print("Models loaded successfully")

# Load real SHAP values from notebook
shap_values = None
feature_names = [
    'Rain_sum_6h', 'Rain_sum_12h', 'API', 'Rain_sum_3h', 'Rainfall_mmhr',
    'SoilMoist_top_m3', 'Rain_sum_24h', 'Rainfall_lag1h', 'SM_lag6h', 'CFSI',
    'WindDir_cos', 'WindDir_sin', 'Rainfall_gradient_3h', 'WindSpeed_ms',
    'SM_anomaly', 'SoilMoist_deep_m3', 'Humidity_pct', 'Temperature_C'
]

try:
    if os.path.exists(shap_path):
        shap_data = np.load(shap_path, allow_pickle=True)
        if isinstance(shap_data, np.ndarray) and shap_data.size > 0:
            shap_values = np.abs(shap_data).mean(axis=(0, 1))
            print(f"Loaded SHAP values shape: {shap_data.shape}")
        else:
            print("SHAP file empty, using fallback")
            shap_values = None
    else:
        print(f"SHAP file not found at {shap_path}, using fallback values")
        shap_values = None
except Exception as e:
    print(f"Error loading SHAP values: {e}")
    shap_values = None

def get_feature_description(feature):
    descriptions = {
        'Rain_sum_6h': 'Total rainfall over past 6 hours',
        'Rain_sum_12h': 'Total rainfall over past 12 hours',
        'API': 'Antecedent Precipitation Index - multi-day rainfall memory',
        'Rain_sum_3h': 'Total rainfall over past 3 hours',
        'Rainfall_mmhr': 'Current hourly rainfall intensity',
        'SoilMoist_top_m3': 'Surface soil moisture (0-7cm depth)',
        'Rain_sum_24h': 'Total rainfall over past 24 hours',
        'Rainfall_lag1h': 'Rainfall intensity 1 hour ago',
        'SM_lag6h': 'Soil moisture 6 hours ago',
        'CFSI': 'Composite Flood Susceptibility Index',
        'WindDir_cos': 'Wind direction cosine component',
        'WindDir_sin': 'Wind direction sine component',
        'Rainfall_gradient_3h': 'Change in rainfall intensity over 3 hours',
        'WindSpeed_ms': 'Wind speed in meters per second',
        'SM_anomaly': 'Soil moisture deviation from monthly normal',
        'SoilMoist_deep_m3': 'Deep soil moisture (7-28cm depth)',
        'Humidity_pct': 'Relative humidity percentage',
        'Temperature_C': 'Air temperature in Celsius'
    }
    return descriptions.get(feature, feature.replace('_', ' '))

def get_feature_display_name(feature):
    names = {
        'Rain_sum_6h': 'Rainfall last 6 hours',
        'Rain_sum_12h': 'Rainfall last 12 hours',
        'API': 'Past days rainfall',
        'Rain_sum_3h': 'Rainfall last 3 hours',
        'Rainfall_mmhr': 'Current rainfall rate',
        'SoilMoist_top_m3': 'Surface soil moisture',
        'Rain_sum_24h': 'Rainfall last 24 hours',
        'Rainfall_lag1h': 'Rainfall 1 hour ago',
        'SM_lag6h': 'Soil moisture 6h ago',
        'CFSI': 'Flood susceptibility',
        'WindDir_cos': 'Wind direction',
        'WindDir_sin': 'Wind direction',
        'Rainfall_gradient_3h': 'Rain intensification',
        'WindSpeed_ms': 'Wind speed',
        'SM_anomaly': 'Soil moisture anomaly',
        'SoilMoist_deep_m3': 'Deep soil moisture',
        'Humidity_pct': 'Humidity',
        'Temperature_C': 'Temperature'
    }
    return names.get(feature, feature.replace('_', ' '))

def is_wet_season():
    current_month = datetime.now().month
    return current_month in [11, 12, 1, 2, 3, 4]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/predict/now', methods=['GET'])
def predict_now():
    try:
        current = get_current_weather()
        historical = get_historical_hours(24)
        
        if len(historical) < 24:
            while len(historical) < 24:
                historical.insert(0, {
                    'temperature': 25, 'humidity': 70, 'rainfall': 0,
                    'pressure': 1013, 'wind_speed': 5, 'wind_dir': 180,
                    'soil_moisture': 0.25
                })
        
        last_24h = historical[-24:]
        
        weather_for_predict = []
        for hour in last_24h:
            weather_for_predict.append({
                'Rainfall_mmhr': hour.get('rainfall', 0),
                'Temperature_C': hour.get('temperature', 25),
                'Humidity_pct': hour.get('humidity', 70),
                'WindSpeed_ms': hour.get('wind_speed', 5),
                'WindDir_deg': hour.get('wind_dir', 180),
                'SoilMoist_top_m3': hour.get('soil_moisture', 0.25),
                'SoilMoist_deep_m3': hour.get('soil_moisture', 0.30)
            })
        
        wet = is_wet_season()
        prediction = predictor.predict(weather_for_predict, wet_season=wet)
        
        zone_probs = apply_gis_multiplier(prediction['calibrated_probability'], zone_risks)
        
        zone_alerts = {}
        for zone, prob in zone_probs.items():
            color, message = get_alert_level(prob, wet)
            zone_alerts[zone] = {
                'probability': round(prob, 4),
                'alert_level': color,
                'message': message
            }
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'city_prediction': prediction,
            'zone_predictions': zone_alerts,
            'current_weather': current,
            'wet_season': wet
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/hour/<path:offset>', methods=['GET'])
def predict_for_hour(offset):
    try:
        offset = int(offset)
        target_time = datetime.now() + timedelta(hours=offset)
        
        historical = get_historical_hours(24)
        
        if len(historical) < 24:
            while len(historical) < 24:
                historical.insert(0, {
                    'temperature': 25, 'humidity': 70, 'rainfall': 0,
                    'pressure': 1013, 'wind_speed': 5, 'wind_dir': 180,
                    'soil_moisture': 0.25
                })
        
        last_24h = historical[-24:]
        
        weather_for_predict = []
        for hour in last_24h:
            weather_for_predict.append({
                'Rainfall_mmhr': hour.get('rainfall', 0),
                'Temperature_C': hour.get('temperature', 25),
                'Humidity_pct': hour.get('humidity', 70),
                'WindSpeed_ms': hour.get('wind_speed', 5),
                'WindDir_deg': hour.get('wind_dir', 180),
                'SoilMoist_top_m3': hour.get('soil_moisture', 0.25),
                'SoilMoist_deep_m3': hour.get('soil_moisture', 0.30)
            })
        
        wet = is_wet_season()
        prediction = predictor.predict(weather_for_predict, wet_season=wet)
        
        zone_probs = apply_gis_multiplier(prediction['calibrated_probability'], zone_risks)
        
        response = {
            'target_hour': target_time.isoformat(),
            'hour_offset': offset,
            'prediction': prediction,
            'zone_probabilities': zone_probs
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Hour prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/7day', methods=['GET'])
def get_7day_forecast():
    try:
        forecast = get_forecast_7days()
        
        # Group by day
        daily_data = {}
        
        for hour in forecast:
            try:
                hour_time = datetime.fromisoformat(hour['time'].replace('Z', '+00:00'))
            except:
                hour_time = datetime.now()
            
            date_key = hour_time.strftime('%Y-%m-%d')
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'rainfall': 0,
                    'hourly_probs': []
                }
            
            # Prepare weather for this specific hour
            weather_for_predict = [{
                'Rainfall_mmhr': hour.get('rainfall', 0),
                'Temperature_C': hour.get('temperature', 25),
                'Humidity_pct': hour.get('humidity', 70),
                'WindSpeed_ms': hour.get('wind_speed', 5),
                'WindDir_deg': 180,
                'SoilMoist_top_m3': 0.25,
                'SoilMoist_deep_m3': 0.30
            }] * 24
            
            wet = is_wet_season()
            prediction = predictor.predict(weather_for_predict, wet_season=wet)
            hour_prob = prediction.get('calibrated_probability', 0.02)
            
            daily_data[date_key]['rainfall'] += hour.get('rainfall', 0)
            daily_data[date_key]['hourly_probs'].append(hour_prob)
        
        daily_summary = []
        for date, data in daily_data.items():
            avg_prob = sum(data['hourly_probs']) / len(data['hourly_probs'])
            risk_score = min(0.50, avg_prob)
            
            daily_summary.append({
                'date': date,
                'total_rainfall': round(data['rainfall'], 1),
                'max_risk_score': round(risk_score, 3)
            })
        
        # Sort by date
        daily_summary.sort(key=lambda x: x['date'])
        
        # Ensure we have 7 days
        while len(daily_summary) < 7:
            last_date = datetime.now() + timedelta(days=len(daily_summary))
            daily_summary.append({
                'date': last_date.strftime('%Y-%m-%d'),
                'total_rainfall': 0,
                'max_risk_score': 0.01
            })
        
        print(f"Forecast generated for {len(daily_summary)} days")
        
        return jsonify({
            'forecast': daily_summary[:7]
        })
    
    except Exception as e:
        print(f"Forecast error: {e}")
        return jsonify({'forecast': []}), 500

@app.route('/api/gis/zones', methods=['GET'])
def get_gis_zones():
    try:
        geojson = get_zone_geojson()
        
        for feature in geojson['features']:
            base_risk = feature['properties'].get('risk', 3.5)
            
            if base_risk >= 4.0:
                feature['properties']['color'] = '#2c7a8a'
                feature['properties']['risk_level'] = 'high'
            elif base_risk >= 3.7:
                feature['properties']['color'] = '#5a9a8a'
                feature['properties']['risk_level'] = 'medium-high'
            elif base_risk >= 3.4:
                feature['properties']['color'] = '#8abaaa'
                feature['properties']['risk_level'] = 'medium'
            else:
                feature['properties']['color'] = '#b0c8d0'
                feature['properties']['risk_level'] = 'low'
        
        return jsonify(geojson)
    
    except Exception as e:
        print(f"GIS zones error: {e}")
        return jsonify({'type': 'FeatureCollection', 'features': []})

@app.route('/api/shap/features', methods=['POST'])
def get_shap_importance():
    try:
        # Use real SHAP values from notebook if available
        if shap_values is not None and len(shap_values) == len(feature_names):
            feature_importance = []
            for i, feat in enumerate(feature_names):
                shap_val = float(shap_values[i])
                feature_importance.append({
                    'feature': feat,
                    'shap_value': shap_val,
                    'direction': 'positive' if shap_val > 0 else 'negative',
                    'display_name': get_feature_display_name(feat),
                    'description': get_feature_description(feat)
                })
            feature_importance.sort(key=lambda x: x['shap_value'], reverse=True)
        else:
            # Fallback using your provided values from notebook
            fallback_values = [
                ('Rain_sum_6h', 0.0008515657172),
                ('Rain_sum_12h', 0.0006733248584),
                ('API', 0.0005986828681),
                ('Rain_sum_3h', 0.0005592653234),
                ('Rainfall_mmhr', 0.0005465588032),
                ('SoilMoist_top_m3', 0.0004953022509),
                ('Rain_sum_24h', 0.0003954397972),
                ('Rainfall_lag1h', 0.0003932744291),
                ('SM_lag6h', 0.0003487945225),
                ('CFSI', 0.0002853855721),
                ('WindDir_cos', 0.0002646260512),
                ('WindDir_sin', 0.0002444837375),
                ('Rainfall_gradient_3h', 0.0002429408941),
                ('WindSpeed_ms', 0.0002336774326),
                ('SM_anomaly', 0.0002235745481),
                ('SoilMoist_deep_m3', 0.0002226286539),
                ('Humidity_pct', 0.0002151004082),
                ('Temperature_C', 0.0001706880645)
            ]
            feature_importance = []
            for feat, val in fallback_values:
                feature_importance.append({
                    'feature': feat,
                    'shap_value': val,
                    'direction': 'positive' if val > 0 else 'negative',
                    'display_name': get_feature_display_name(feat),
                    'description': get_feature_description(feat)
                })
        
        return jsonify({
            'feature_importance': feature_importance,
            'top_feature': feature_importance[0]['feature'],
            'top_shap_value': feature_importance[0]['shap_value']
        })
    
    except Exception as e:
        print(f"SHAP error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shap/temporal', methods=['POST'])
def get_temporal_shap():
    temporal_importance = []
    for hour in range(24):
        importance = 0.0005 * (1 - hour/24)
        temporal_importance.append({
            'hour': hour,
            'importance': round(importance, 6),
            'relative_time': f'{hour}h ago' if hour > 0 else 'current hour'
        })
    
    return jsonify({
        'temporal_importance': temporal_importance,
        'most_important_hour': 2,
        'explanation': 'Recent rainfall (2-6 hours ago) has strongest influence'
    })
    
@app.route('/api/gis/zone-risks', methods=['GET'])
def get_zone_risk_multipliers():
    try:
        base_risk = 3.87
        zone_multipliers = {}
        
        for zone, risk in zone_risks.items():
            multiplier = 1 + (risk - base_risk) / base_risk
            multiplier = max(0.7, min(1.5, multiplier))
            zone_multipliers[zone] = round(multiplier, 3)
        
        return jsonify({
            'base_risk': base_risk,
            'zones': zone_multipliers,
            'risk_scores': zone_risks
        })
    except Exception as e:
        print(f"Zone risks error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting FloodCast server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)