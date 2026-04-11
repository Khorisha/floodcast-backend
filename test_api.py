from utils.predictor import FloodPredictor
import os

model_path = 'models/final_model.tflite'
scaler_path = 'models/scaler.pkl'
iso_path = 'models/gru_iso.pkl'

print("Testing model loading...")

try:
    predictor = FloodPredictor(model_path, scaler_path, iso_path)
    print("Predictor created successfully")
    
    # Test with dummy data
    dummy_data = [{
        'Rainfall_mmhr': 0,
        'Temperature_C': 25,
        'Humidity_pct': 70,
        'WindSpeed_ms': 5,
        'WindDir_deg': 180,
        'SoilMoist_top_m3': 0.25,
        'SoilMoist_deep_m3': 0.30
    }] * 24
    
    result = predictor.predict(dummy_data, wet_season=True)
    print(f"Prediction result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()