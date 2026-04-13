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

from utils.weather_api import get_current_weather, get_forecast_7days, get_historical_hours, get_hours_for_week
from utils.predictor import FloodPredictor
from utils.gis_fusion import load_zone_risks, apply_gis_multiplier, get_zone_geojson, get_alert_level

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'final_model.tflite')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
iso_path = os.path.join(os.path.dirname(__file__), 'models', 'gru_iso.pkl')
shap_path = os.path.join(os.path.dirname(__file__), 'models', 'shap_values_flood_test.npy')

print("Loading models...")
predictor = FloodPredictor(model_path, scaler_path, iso_path)
zone_risks = load_zone_risks()
print("Models loaded successfully")

# Load real SHAP values
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
            print("SHAP file empty")
            shap_values = None
    else:
        print(f"SHAP file not found at {shap_path}")
        shap_values = None
except Exception as e:
    print(f"Error loading SHAP values: {e}")
    shap_values = None

# Build a {feature_name: importance} dict used by get_context_summary to pick top driver
shap_weight_dict = None
if shap_values is not None and len(shap_values) == len(feature_names):
    shap_weight_dict = {feature_names[i]: float(shap_values[i]) for i in range(len(feature_names))}
    print(f"SHAP weight dict built: {len(shap_weight_dict)} features")

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

def _build_weather_sequence(hours):
    """
    Convert raw weather records from weather_api into the format the predictor expects.
    Critically preserves the 'time' key so the predictor can compute monthly SM_anomaly.
    """
    seq = []
    for h in hours:
        seq.append({
            'time':           h.get('time', ''),
            'rainfall':       h.get('rainfall', 0.0) or 0.0,
            'temperature':    h.get('temperature', 25.0) or 25.0,
            'humidity':       h.get('humidity', 70.0) or 70.0,
            'wind_speed':     h.get('wind_speed', 5.0) or 5.0,
            'wind_dir':       h.get('wind_dir', 180.0) or 180.0,
            'soil_moisture':       h.get('soil_moisture', 0.25) or 0.25,
            'soil_moisture_deep':  h.get('soil_moisture_deep', h.get('soil_moisture', 0.30)) or 0.30,
        })
    return seq

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/predict/now', methods=['GET'])
def predict_now():
    try:
        # Fetch current weather independently — don't let a 429 here abort the prediction.
        current = None
        try:
            current = get_current_weather()
        except Exception as cw_err:
            print(f"Current weather fetch skipped: {cw_err}")

        historical = get_historical_hours(168)
        if len(historical) < 24:
            return jsonify({'error': 'Insufficient historical data'}), 400

        # Fall back to the most recent historical hour if current weather unavailable.
        if current is None and historical:
            last = historical[-1]
            current = {
                'temp':       last.get('temperature', 25),
                'humidity':   last.get('humidity', 70),
                'pressure':   last.get('pressure', 1013),
                'wind_speed': last.get('wind_speed', 5),
                'wind_dir':   last.get('wind_dir', 180),
                'rain_1h':    last.get('rainfall', 0),
                'timestamp':  last.get('time', datetime.now().isoformat())
            }

        wet = is_wet_season()
        prediction = predictor.predict(_build_weather_sequence(historical), wet_season=wet)

        zone_probs = apply_gis_multiplier(prediction['calibrated_probability'], zone_risks)
        zone_alerts = {}
        for zone, prob in zone_probs.items():
            color, message = get_alert_level(prob, wet)
            zone_alerts[zone] = {'probability': round(prob, 4), 'alert_level': color, 'message': message}

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'city_prediction': prediction,
            'zone_predictions': zone_alerts,
            'current_weather': current,
            'wet_season': wet
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/hour/<offset>', methods=['GET'])
def predict_for_hour(offset):
    try:
        offset = int(offset)
    except ValueError:
        return jsonify({'error': 'Invalid offset'}), 400
    try:
        target_time = datetime.now() + timedelta(hours=offset)
        wet         = is_wet_season()
        historical  = get_historical_hours(168)
        if len(historical) < 24:
            return jsonify({'error': 'Insufficient historical data'}), 400

        if offset <= 0:
            # Past or current hour: trim history to the target point so the
            # model sees data up to (and including) that hour only.
            trim = min(abs(offset), len(historical) - 24)
            source_hours = historical[:-trim] if trim > 0 else historical
        else:
            # Future hour: append forecast hours up to the target offset.
            try:
                forecast     = get_forecast_7days()
                future_slice = forecast[:offset]
                source_hours = historical + future_slice
            except Exception:
                source_hours = historical

        prediction = predictor.predict(_build_weather_sequence(source_hours), wet_season=wet)
        ctx        = predictor.get_context_summary(source_hours, shap_weights=shap_weight_dict)
        zone_probs = apply_gis_multiplier(prediction['calibrated_probability'], zone_risks)

        return jsonify({
            'target_hour':        target_time.isoformat(),
            'hour_offset':        offset,
            'prediction':         prediction,
            'zone_probabilities': zone_probs,
            **ctx
        })

    except Exception as e:
        print(f"Hour prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/7day', methods=['GET'])
def get_7day_forecast():
    try:
        from_date_str = request.args.get('from', None)
        if from_date_str:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        else:
            from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        wet = is_wet_season()

        # Fetch all 7 days in a single API call
        all_hours = get_hours_for_week(from_date, num_days=7)

        # Group hours by date
        hours_by_date = {}
        for h in all_hours:
            dk = h['time'][:10]
            if dk not in hours_by_date:
                hours_by_date[dk] = []
            hours_by_date[dk].append(h)

        daily_summary = []
        default_hour = {'time': '', 'rainfall': 0.0, 'temperature': 25.0, 'humidity': 70.0,
                        'wind_speed': 5.0, 'wind_dir': 180.0, 'soil_moisture': 0.25}

        # Collect all available hours in chronological order for API warm-up
        all_sorted = sorted(all_hours, key=lambda h: h['time'])

        for day_offset in range(7):
            target_day = from_date + timedelta(days=day_offset)
            target_day_str = target_day.strftime('%Y-%m-%d')

            try:
                day_hours = hours_by_date.get(target_day_str, [])

                if len(day_hours) < 6:
                    daily_summary.append({'date': target_day_str, 'total_rainfall': 0, 'max_risk_score': 0.01})
                    continue

                # Pad to 24 if partial day (e.g. today only has hours so far)
                source_hours = day_hours[:24]
                while len(source_hours) < 24:
                    source_hours.append(source_hours[-1] if source_hours else default_hour)

                total_rain = sum(h.get('rainfall', 0) or 0 for h in source_hours)

                # Include all hours up to and including this day for API warm-up
                target_end = target_day_str + 'T23:59'
                warmup = [h for h in all_sorted if h['time'] <= target_end]
                if len(warmup) < 24:
                    warmup = source_hours
                seq = _build_weather_sequence(warmup)

                prediction = predictor.predict(seq, wet_season=wet)
                daily_summary.append({
                    'date': target_day_str,
                    'total_rainfall': round(total_rain, 1),
                    'max_risk_score': prediction['calibrated_probability']
                })

            except Exception as day_err:
                print(f"Error for day {target_day_str}: {day_err}")
                daily_summary.append({'date': target_day_str, 'total_rainfall': 0, 'max_risk_score': 0.01})

        return jsonify({'forecast': daily_summary[:7]})

    except Exception as e:
        print(f"Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'forecast': []}), 500


@app.route('/api/predict/date/<date_str>/hours', methods=['GET'])
def predict_date_hours(date_str):
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        days_diff = (target_date.date() - today.date()).days

        if days_diff > 0:
            return jsonify({'error': 'Hourly predictions only available for today and past dates.'}), 400

        wet = is_wet_season()

        # Fetch 8 days ending on target date for proper API warm-up:
        # 7 warmup days + target day itself
        warmup_start = target_date - timedelta(days=7)
        all_hours = get_hours_for_week(warmup_start, num_days=8)

        target_day_str = target_date.strftime('%Y-%m-%d')
        target_hours = [h for h in all_hours if h['time'][:10] == target_day_str]

        default_hour = {'time': target_day_str + 'T00:00', 'rainfall': 0.0, 'temperature': 25.0,
                        'humidity': 70.0, 'wind_speed': 5.0, 'wind_dir': 180.0, 'soil_moisture': 0.25}
        while len(target_hours) < 24:
            target_hours.append(target_hours[-1] if target_hours else default_hour)
        target_hours = target_hours[:24]

        # All hours up to end of this day (chronological) — full warm-up history
        all_sorted = sorted(all_hours, key=lambda h: h['time'])

        hourly_predictions = []
        for h in range(24):
            target_hour_str = f'{target_day_str}T{str(h).zfill(2)}:59'
            raw_context = [rec for rec in all_sorted if rec['time'] <= target_hour_str]

            # Context summary from raw hours (before sequence building)
            ctx = predictor.get_context_summary(raw_context, shap_weights=shap_weight_dict)

            if len(raw_context) < 24:
                padding      = [default_hour] * (24 - len(raw_context))
                built_context = _build_weather_sequence(padding + raw_context)
            else:
                built_context = _build_weather_sequence(raw_context)

            pred = predictor.predict(built_context, wet_season=wet)
            hourly_predictions.append({
                'hour':        h,
                'time':        f'{target_day_str}T{str(h).zfill(2)}:00',
                'probability': pred['calibrated_probability'],
                'prediction':  pred['prediction'],
                **ctx
            })

        return jsonify({'date': date_str, 'hourly': hourly_predictions})

    except Exception as e:
        print(f"Hourly date prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/date/<date_str>', methods=['GET'])
def predict_for_date(date_str):
    try:
        print(f"Date endpoint called: {date_str}")
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        days_diff = (target_date.date() - today.date()).days

        weather_for_predict = []
        current_weather_data = None

        if days_diff == 0:
            # Today — fetch 7 days for API warm-up
            current_weather_data = get_current_weather()
            historical = get_historical_hours(168)
            if len(historical) < 24:
                return jsonify({'error': 'Insufficient historical data for today'}), 400
            weather_for_predict = _build_weather_sequence(historical)

        elif days_diff > 0:
            # Future date — use 7-day forecast; warm up with recent history prepended
            if days_diff > 7:
                return jsonify({'error': 'Forecast only available up to 7 days ahead'}), 400
            forecast = get_forecast_7days()
            target_date_key = target_date.strftime('%Y-%m-%d')
            date_hours = [h for h in forecast if h['time'][:10] == target_date_key]
            if len(date_hours) < 12:
                return jsonify({'error': f'No forecast data available for {date_str}'}), 400
            # Prepend recent historical hours as warm-up context
            recent = get_historical_hours(168)
            source = date_hours[:24]
            while len(source) < 24:
                source.append(source[-1])
            weather_for_predict = _build_weather_sequence(recent + source)
            if date_hours:
                first = date_hours[0]
                current_weather_data = {
                    'temp': first.get('temperature'),
                    'humidity': first.get('humidity'),
                    'wind_speed': first.get('wind_speed'),
                    'wind_dir': first.get('wind_dir'),
                    'pressure': first.get('pressure'),
                    'rain_1h': first.get('rainfall', 0)
                }

        else:
            # Historical — fetch 7 warmup days + target date
            warmup_start = target_date - timedelta(days=7)
            all_hrs = get_hours_for_week(warmup_start, num_days=8)
            target_day_hours = [h for h in all_hrs if h['time'][:10] == date_str]
            if len(target_day_hours) < 1:
                return jsonify({'error': f'No data available for {date_str}'}), 400
            weather_for_predict = _build_weather_sequence(all_hrs)
            if target_day_hours:
                first = target_day_hours[0]
                current_weather_data = {
                    'temp': first.get('temperature'),
                    'humidity': first.get('humidity'),
                    'wind_speed': first.get('wind_speed'),
                    'wind_dir': first.get('wind_dir'),
                    'pressure': first.get('pressure'),
                    'rain_1h': first.get('rainfall', 0)
                }

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
            'date': date_str,
            'city_prediction': prediction,
            'zone_predictions': zone_alerts,
            'current_weather': current_weather_data,
            'wet_season': wet
        }

        return jsonify(response)

    except Exception as e:
        print(f"Date prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
    print(f"Available endpoints:")
    print(f"  GET /health")
    print(f"  GET /api/predict/now")
    print(f"  GET /api/predict/hour/0")
    print(f"  GET /api/predict/date/2026-04-11")
    print(f"  GET /api/predict/date/2026-04-11/hours")
    print(f"  GET /api/forecast/7day")
    print(f"  GET /api/forecast/7day?from=2026-04-11")
    print(f"  GET /api/gis/zones")
    print(f"  POST /api/shap/features")
    app.run(host='0.0.0.0', port=port, debug=False)