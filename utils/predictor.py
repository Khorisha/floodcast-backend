import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
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
    
    def prepare_features(self, weather_data):
        try:
            # Handle both key naming conventions (API vs internal)
            rainfall = weather_data.get('rainfall', weather_data.get('Rainfall_mmhr', 0))
            temp = weather_data.get('temperature', weather_data.get('Temperature_C', 25))
            humidity = weather_data.get('humidity', weather_data.get('Humidity_pct', 70))
            wind_speed = weather_data.get('wind_speed', weather_data.get('WindSpeed_ms', 5))
            wind_dir = weather_data.get('wind_dir', weather_data.get('WindDir_deg', 180))
            soil_top = weather_data.get('soil_moisture', weather_data.get('SoilMoist_top_m3', 0.25))
            soil_deep = weather_data.get('soil_moisture_deep', weather_data.get('SoilMoist_deep_m3', 0.30))
            
            df = pd.DataFrame([{
                'Rainfall_mmhr': rainfall,
                'Temperature_C': temp,
                'Humidity_pct': humidity,
                'WindSpeed_ms': wind_speed,
                'WindDir_deg': wind_dir,
                'SoilMoist_top_m3': soil_top,
                'SoilMoist_deep_m3': soil_deep
            }])
            
            df['WindDir_sin'] = np.sin(2 * np.pi * df['WindDir_deg'] / 360)
            df['WindDir_cos'] = np.cos(2 * np.pi * df['WindDir_deg'] / 360)
            
            df['API'] = rainfall
            
            for window in [3, 6, 12, 24]:
                df[f'Rain_sum_{window}h'] = rainfall * window
            
            df['PWI'] = (df['Humidity_pct'] / 100) * (df['Temperature_C'] / 25)
            df['SoilSaturation'] = df['SoilMoist_top_m3'] / 0.4
            df['CFSI'] = df['PWI'] * df['SoilSaturation']
            
            df['Rainfall_gradient_3h'] = rainfall
            df['Rainfall_lag1h'] = 0
            
            df['SM_anomaly'] = df['SoilMoist_top_m3'] - 0.2
            df['SM_lag6h'] = df['SoilMoist_top_m3']
            
            result = df[self.feature_names].values[0]
            return result
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return np.zeros(len(self.feature_names))
    
    def create_sequence(self, last_24_hours):
        sequence = []
        for hour_data in last_24_hours:
            features = self.prepare_features(hour_data)
            sequence.append(features)
        
        sequence_array = np.array(sequence, dtype=np.float32)
        
        if self.scaler is not None:
            try:
                scaled_sequence = self.scaler.transform(sequence_array)
            except Exception as e:
                print(f"Scaling error: {e}")
                scaled_sequence = sequence_array
        else:
            scaled_sequence = sequence_array
        
        model_input = scaled_sequence.reshape(1, 24, len(self.feature_names))
        return model_input
    
    def predict(self, last_24_hours, wet_season=True):
        try:
            if len(last_24_hours) < 24:
                print(f"Warning: Only {len(last_24_hours)} hours provided, need 24")
                # Pad with dummy data if needed
                while len(last_24_hours) < 24:
                    last_24_hours.insert(0, {
                        'rainfall': 0, 'temperature': 25, 'humidity': 70,
                        'wind_speed': 5, 'wind_dir': 180, 'soil_moisture': 0.25
                    })
            
            model_input = self.create_sequence(last_24_hours)
            
            if self.interpreter is not None:
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], model_input)
                self.interpreter.invoke()
                raw_prob = self.interpreter.get_tensor(output_details[0]['index'])[0][0]
            else:
                raw_prob = 0.05
            
            if self.iso is not None:
                try:
                    calibrated_prob = self.iso.predict([[raw_prob]])[0]
                except Exception as e:
                    print(f"Iso prediction error: {e}")
                    calibrated_prob = raw_prob
            else:
                calibrated_prob = raw_prob
            
            calibrated_prob = max(0.01, min(0.99, calibrated_prob))
            
            if wet_season:
                threshold = 0.02
            else:
                threshold = 0.05
            
            prediction = 1 if calibrated_prob >= threshold else 0
            
            return {
                'raw_probability': float(raw_prob),
                'calibrated_probability': float(calibrated_prob),
                'prediction': int(prediction),
                'threshold_used': threshold,
                'wet_season': wet_season
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'raw_probability': 0.05,
                'calibrated_probability': 0.05,
                'prediction': 0,
                'threshold_used': 0.02,
                'wet_season': wet_season,
                'error': str(e)
            }