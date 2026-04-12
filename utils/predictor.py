import numpy as np
import tensorflow as tf
import pickle
import os


class FloodPredictor:
    def __init__(self, model_path, scaler_path, iso_path):
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        print(f"Loading iso from: {iso_path}")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            print("TFLite model loaded successfully")
        except Exception as e:
            print(f"Error loading TFLite: {e}")
            self.interpreter = None

        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None

        try:
            with open(iso_path, 'rb') as f:
                self.iso = pickle.load(f)
            print("Isotonic calibrator loaded successfully")
        except Exception as e:
            print(f"Error loading iso: {e}")
            self.iso = None

        self.feature_names = [
            'Rainfall_mmhr', 'Temperature_C', 'Humidity_pct', 'WindSpeed_ms',
            'WindDir_sin', 'WindDir_cos', 'SoilMoist_top_m3', 'SoilMoist_deep_m3',
            'SM_anomaly', 'SM_lag6h', 'API', 'Rain_sum_3h', 'Rain_sum_6h',
            'Rain_sum_12h', 'Rain_sum_24h', 'CFSI', 'Rainfall_gradient_3h',
            'Rainfall_lag1h'
        ]

        # ERA5 monthly mean surface soil moisture for Port Louis (2010-2024)
        self.MONTHLY_SM_CLIM = {
            1: 0.298, 2: 0.292, 3: 0.284, 4: 0.270,
            5: 0.254, 6: 0.239, 7: 0.231, 8: 0.227,
            9: 0.231, 10: 0.241, 11: 0.257, 12: 0.276
        }

        # (dynamic range scaler removed — isotonic calibrator handles calibration)

        # Human-readable names for SHAP feature display
        self.FEATURE_DISPLAY = {
            'Rain_sum_6h':          '6-hour rainfall total',
            'Rain_sum_12h':         '12-hour rainfall total',
            'API':                  'Multi-day rainfall buildup',
            'Rain_sum_3h':          '3-hour rainfall burst',
            'Rainfall_mmhr':        'Current rainfall intensity',
            'SoilMoist_top_m3':     'Saturated surface soil',
            'Rain_sum_24h':         '24-hour rainfall total',
            'Rainfall_lag1h':       'Rainfall one hour ago',
            'SM_lag6h':             'Persistently wet soil',
            'CFSI':                 'Compound flood susceptibility',
            'WindDir_cos':          'Wind direction',
            'WindDir_sin':          'Wind direction',
            'Rainfall_gradient_3h': 'Rapidly intensifying rain',
            'WindSpeed_ms':         'Wind conditions',
            'SM_anomaly':           'Above-normal soil moisture',
            'SoilMoist_deep_m3':    'Saturated deep soil',
            'Humidity_pct':         'High atmospheric moisture',
            'Temperature_C':        'Temperature',
        }

    def _scale_model_output(self, raw_prob):
        """Pass raw model output through unchanged — isotonic calibration handles the mapping."""
        return max(0.0, min(0.99, raw_prob))

    def _get_rainfall_intensity_factor(self, hours):
        rainfall = []
        for h in hours[-24:]:
            r = h.get('rainfall', 0)
            if r is not None:
                rainfall.append(float(r))
        if not rainfall:
            return None
        peak_1h = max(rainfall[-1:])  if len(rainfall) >= 1 else 0.0
        peak_3h = max(rainfall[-3:])  if len(rainfall) >= 3 else peak_1h
        peak_6h = max(rainfall[-6:])  if len(rainfall) >= 6 else peak_3h
        risk = 0.0
        if   peak_1h >= 15: risk = 0.80
        elif peak_1h >= 10: risk = 0.50
        elif peak_1h >=  5: risk = 0.25
        elif peak_1h >=  2: risk = 0.10
        if peak_3h >= 25 and risk < 0.50: risk = 0.50
        elif peak_3h >= 15 and risk < 0.30: risk = 0.30
        if peak_6h >= 40 and risk < 0.70: risk = 0.70
        return risk if risk > 0 else None

    def get_context_summary(self, hours, shap_weights=None):
        """
        Return per-hour context metrics and the dominant flood driver.
        shap_weights: dict {feature_name: importance_score} built from pre-computed SHAP values.
        When provided, the top driver is determined by SHAP-weighted feature activation.
        """
        if not hours:
            return {}
        recent = hours[-24:] if len(hours) >= 24 else hours

        def _corrected(h):
            r = float(h.get('rainfall', 0) or 0)
            return r * 1.6 if r >= 2.0 else r

        rain      = [_corrected(h) for h in recent]
        soil      = [float(h.get('soil_moisture', 0.25) or 0.25) for h in recent]
        last      = recent[-1]
        soil_deep = float(last.get('soil_moisture_deep', soil[-1] * 0.9) or soil[-1] * 0.9)
        last_h    = float(last.get('humidity', 70) or 70)
        last_t    = float(last.get('temperature', 25) or 25)

        peak_1h  = rain[-1] if rain else 0.0
        rain_3h  = sum(rain[-3:])
        rain_6h  = sum(rain[-6:])
        rain_12h = sum(rain[-12:])
        rain_24h = sum(rain)
        soil_avg = sum(soil[-6:]) / max(len(soil[-6:]), 1)

        api_val = 0.0
        for r in rain:
            api_val = r + 0.90 * api_val

        FIELD_CAP = 0.4
        mean_t = sum(float(h.get('temperature', 25) or 25) for h in recent) / max(len(recent), 1)
        soil_sat = min(soil_avg / FIELD_CAP, 1.0)
        api_norm = min(api_val / (api_val + 1e-8), 1.0)
        cfsi     = api_norm * soil_sat * (last_h / 100.0) * (last_t / mean_t if mean_t else 1.0)

        rain_lag1h   = rain[-2] if len(rain) >= 2 else 0.0
        rain_grad_3h = abs(rain[-1] - rain[-4]) if len(rain) >= 4 else 0.0

        time_str = last.get('time', '')
        month = 4
        if time_str and len(time_str) >= 7:
            try:
                month = int(time_str[5:7])
            except (ValueError, IndexError):
                pass
        sm_anomaly = abs(soil_avg - self.MONTHLY_SM_CLIM.get(month, 0.260))

        # Normalize each feature to 0-1 based on physically meaningful ranges
        feature_activations = {
            'Rain_sum_6h':          min(rain_6h   / 30.0,  1.0),
            'Rain_sum_12h':         min(rain_12h  / 50.0,  1.0),
            'API':                  min(api_val   / 25.0,  1.0),
            'Rain_sum_3h':          min(rain_3h   / 20.0,  1.0),
            'Rainfall_mmhr':        min(peak_1h   / 15.0,  1.0),
            'SoilMoist_top_m3':     min(max(soil_avg - 0.20, 0) / 0.18, 1.0),
            'Rain_sum_24h':         min(rain_24h  / 80.0,  1.0),
            'Rainfall_lag1h':       min(rain_lag1h / 10.0, 1.0),
            'SM_lag6h':             min(max(soil_avg - 0.20, 0) / 0.18, 1.0),
            'CFSI':                 min(cfsi,               1.0),
            'Rainfall_gradient_3h': min(rain_grad_3h / 8.0, 1.0),
            'SM_anomaly':           min(sm_anomaly  / 0.08, 1.0),
            'SoilMoist_deep_m3':    min(max(soil_deep - 0.20, 0) / 0.18, 1.0),
            'Humidity_pct':         min(max(last_h - 60, 0) / 40.0, 1.0),
            'WindDir_cos':          0.0,
            'WindDir_sin':          0.0,
            'WindSpeed_ms':         0.0,
            'Temperature_C':        0.0,
        }

        if shap_weights and len(shap_weights) > 0:
            # Weight activation by SHAP importance — picks the feature that is
            # both highly active right now AND historically important for floods
            weighted = {
                feat: feature_activations.get(feat, 0.0) * shap_weights.get(feat, 0.0)
                for feat in feature_activations
            }
            top_feat  = max(weighted, key=weighted.get)
            top_score = weighted[top_feat]
        else:
            # Fallback: rank by activation only
            scores = {k: v for k, v in feature_activations.items()
                      if k not in ('WindDir_cos', 'WindDir_sin', 'WindSpeed_ms', 'Temperature_C')}
            top_feat  = max(scores, key=scores.get)
            top_score = scores[top_feat]

        if top_score < 0.01:
            top_driver = 'No dominant flood driver — background conditions'
        else:
            top_driver = self.FEATURE_DISPLAY.get(top_feat, top_feat)

        return {
            'rainfall_now_mm': round(peak_1h,  1),
            'rain_6h_mm':      round(rain_6h,  1),
            'rain_24h_mm':     round(rain_24h, 1),
            'soil_moisture':   round(soil_avg, 3),
            'top_driver':      top_driver,
        }

    def create_sequence(self, hours):
        """
        Build a (24, 18) feature matrix matching the training notebook exactly.
        hours may contain MORE than 24 entries — extra history warms up API and rolling sums.
        Only the LAST 24 rows are fed to the model.
        """
        n = len(hours)

        def _get(h, *keys):
            for k in keys:
                if k in h:
                    v = h[k]
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
            return None

        rainfall_raw = np.array([_get(h, 'rainfall', 'Rainfall_mmhr') or 0.0 for h in hours], dtype=np.float64)
        temp         = np.array([_get(h, 'temperature', 'Temperature_C') or 25.0 for h in hours], dtype=np.float64)
        humidity     = np.array([_get(h, 'humidity', 'Humidity_pct') or 70.0 for h in hours], dtype=np.float64)
        wind_speed   = np.array([_get(h, 'wind_speed', 'WindSpeed_ms') or 5.0 for h in hours], dtype=np.float64)
        wind_dir     = np.array([_get(h, 'wind_dir', 'WindDir_deg') or 180.0 for h in hours], dtype=np.float64)
        soil_top     = np.array([_get(h, 'soil_moisture', 'SoilMoist_top_m3') or 0.25 for h in hours], dtype=np.float64)
        soil_deep    = np.array([_get(h, 'soil_moisture_deep', 'SoilMoist_deep_m3') or 0.30 for h in hours], dtype=np.float64)

        rainfall = np.where(rainfall_raw >= 2.0, rainfall_raw * 1.6, rainfall_raw)

        wind_sin = np.sin(2 * np.pi * wind_dir / 360.0)
        wind_cos = np.cos(2 * np.pi * wind_dir / 360.0)

        K_DECAY = 0.90
        api = np.zeros(n)
        api[0] = rainfall[0]
        for i in range(1, n):
            api[i] = rainfall[i] + K_DECAY * api[i - 1]

        def rolling_sum(arr, w):
            out = np.zeros(len(arr))
            for i in range(len(arr)):
                out[i] = arr[max(0, i - w + 1):i + 1].sum()
            return out

        rain_sum_3h  = rolling_sum(rainfall, 3)
        rain_sum_6h  = rolling_sum(rainfall, 6)
        rain_sum_12h = rolling_sum(rainfall, 12)
        rain_sum_24h = rolling_sum(rainfall, 24)

        FIELD_CAP = 0.4
        mean_temp = temp.mean() if temp.mean() != 0 else 1.0
        pwi       = (humidity / 100.0) * (temp / mean_temp)
        soil_sat  = np.clip(soil_top / FIELD_CAP, 0.0, 1.0)
        api_range = api.max() - api.min()
        api_norm  = (api - api.min()) / (api_range + 1e-8)
        cfsi      = api_norm * soil_sat * pwi

        rain_grad_3h = np.zeros(n)
        for i in range(3, n):
            rain_grad_3h[i] = rainfall[i] - rainfall[i - 3]

        rain_lag1h = np.zeros(n)
        rain_lag1h[1:] = rainfall[:-1]

        sm_lag6h = np.empty(n)
        sm_lag6h[:6] = soil_top[0]
        sm_lag6h[6:] = soil_top[:-6]

        sm_anomaly = np.zeros(n)
        for i, h in enumerate(hours):
            time_str = h.get('time', '')
            month = 4
            if time_str and len(time_str) >= 7:
                try:
                    month = int(time_str[5:7])
                except (ValueError, IndexError):
                    pass
            clim_mean = self.MONTHLY_SM_CLIM.get(month, 0.260)
            sm_anomaly[i] = soil_top[i] - clim_mean

        full_sequence = np.column_stack([
            rainfall, temp, humidity, wind_speed, wind_sin, wind_cos,
            soil_top, soil_deep, sm_anomaly, sm_lag6h, api,
            rain_sum_3h, rain_sum_6h, rain_sum_12h, rain_sum_24h,
            cfsi, rain_grad_3h, rain_lag1h
        ]).astype(np.float32)

        sequence = full_sequence[-24:]

        if self.scaler is not None:
            try:
                scaled_sequence = self.scaler.transform(sequence)
            except Exception as e:
                print(f"Scaling error: {e}")
                scaled_sequence = sequence
        else:
            scaled_sequence = sequence

        return scaled_sequence.reshape(1, 24, len(self.feature_names))

    def predict(self, hours, wet_season=True):
        try:
            if len(hours) < 24:
                print(f"Warning: Only {len(hours)} hours provided, padding to 24")
                pad = {'rainfall': 0.0, 'temperature': 25.0, 'humidity': 70.0,
                       'wind_speed': 5.0, 'wind_dir': 180.0, 'soil_moisture': 0.25, 'time': ''}
                while len(hours) < 24:
                    hours.insert(0, pad)

            physics_risk = self._get_rainfall_intensity_factor(hours)
            model_input  = self.create_sequence(hours)

            if self.interpreter is not None:
                input_details  = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                self.interpreter.set_tensor(input_details[0]['index'], model_input)
                self.interpreter.invoke()
                raw_prob = float(self.interpreter.get_tensor(output_details[0]['index'])[0][0])
            else:
                raw_prob = 0.05

            scaled_prob = self._scale_model_output(raw_prob)

            if physics_risk is not None:
                if physics_risk > 0.30:
                    calibrated_prob = physics_risk * 0.7 + scaled_prob * 0.3
                else:
                    calibrated_prob = scaled_prob * 0.6 + physics_risk * 0.4
            else:
                calibrated_prob = scaled_prob

            if self.iso is not None:
                try:
                    iso_prob = float(self.iso.predict([[raw_prob]])[0])
                    calibrated_prob = calibrated_prob * 0.6 + iso_prob * 0.4
                except Exception as e:
                    print(f"Iso prediction error: {e}")

            calibrated_prob *= 1.05 if wet_season else 0.95

            has_rain = any((h.get('rainfall') or 0) > 1.0 for h in hours[-6:])
            if not has_rain:
                calibrated_prob = min(calibrated_prob, 0.05)

            calibrated_prob = max(0.0, min(0.99, calibrated_prob))
            threshold  = 0.02 if wet_season else 0.05
            prediction = 1 if calibrated_prob >= threshold else 0

            print(f"DEBUG — raw: {raw_prob:.4f}  scaled: {scaled_prob:.4f}  "
                  f"physics: {physics_risk}  final: {calibrated_prob:.4f}")

            return {
                'raw_probability':        raw_prob,
                'calibrated_probability': calibrated_prob,
                'prediction':             prediction,
                'threshold_used':         threshold,
                'wet_season':             wet_season
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'raw_probability':        0.05,
                'calibrated_probability': 0.0,
                'prediction':             0,
                'threshold_used':         0.02,
                'wet_season':             wet_season,
                'error':                  str(e)
            }
