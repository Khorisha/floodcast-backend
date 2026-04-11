import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

PORT_LOUIS_LAT = -20.1609
PORT_LOUIS_LON = 57.5012

def get_current_weather():
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': PORT_LOUIS_LAT,
            'longitude': PORT_LOUIS_LON,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            return {
                'temp': current.get('temperature_2m', 25),
                'humidity': current.get('relative_humidity_2m', 70),
                'pressure': current.get('pressure_msl', 1013),
                'wind_speed': current.get('wind_speed_10m', 5),
                'wind_dir': current.get('wind_direction_10m', 180),
                'rain_1h': current.get('precipitation', 0),
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return {
        'temp': 25.0,
        'humidity': 70,
        'pressure': 1013,
        'wind_speed': 5.0,
        'wind_dir': 180,
        'rain_1h': 0,
        'timestamp': datetime.now().isoformat()
    }

def get_forecast_7days():
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': PORT_LOUIS_LAT,
            'longitude': PORT_LOUIS_LON,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m',
            'forecast_days': 7,
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            hourly = data['hourly']
            records = []
            for i in range(len(hourly['time'])):
                records.append({
                    'time': hourly['time'][i],
                    'temperature': hourly['temperature_2m'][i],
                    'humidity': hourly['relative_humidity_2m'][i],
                    'rainfall': hourly['precipitation'][i],
                    'pressure': hourly['pressure_msl'][i],
                    'wind_speed': hourly['wind_speed_10m'][i],
                    'wind_dir': hourly['wind_direction_10m'][i]
                })
            return records
    except Exception as e:
        print(f"Forecast API error: {e}")
    
    return generate_mock_forecast()

def generate_mock_forecast():
    records = []
    start_time = datetime.now()
    for i in range(168):
        hour_time = start_time + timedelta(hours=i)
        records.append({
            'time': hour_time.isoformat(),
            'temperature': 25 + (i % 24) / 10,
            'humidity': 70 + (i % 12) - 6,
            'rainfall': max(0, 2 + np.sin(i/12) * 2),
            'pressure': 1013 + (i % 10) - 5,
            'wind_speed': 5 + (i % 8),
            'wind_dir': (i * 15) % 360
        })
    return records

def get_historical_hours(hours_back=24):
    try:
        start = datetime.now() - timedelta(hours=hours_back)
        end = datetime.now()
        
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude': PORT_LOUIS_LAT,
            'longitude': PORT_LOUIS_LON,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            hourly = data['hourly']
            records = []
            for i in range(len(hourly['time'])):
                records.append({
                    'time': hourly['time'][i],
                    'temperature': hourly['temperature_2m'][i],
                    'humidity': hourly['relative_humidity_2m'][i],
                    'rainfall': hourly['precipitation'][i],
                    'pressure': hourly['pressure_msl'][i],
                    'wind_speed': hourly['wind_speed_10m'][i],
                    'wind_dir': hourly['wind_direction_10m'][i],
                    'soil_moisture': 0.25
                })
            return records
    except Exception as e:
        print(f"Historical API error: {e}")
    
    return generate_mock_historical(hours_back)

def generate_mock_historical(hours_back):
    records = []
    now = datetime.now()
    for i in range(hours_back):
        hour_time = now - timedelta(hours=hours_back - i)
        records.append({
            'time': hour_time.isoformat(),
            'temperature': 25,
            'humidity': 70,
            'rainfall': 0,
            'pressure': 1013,
            'wind_speed': 5,
            'wind_dir': 180,
            'soil_moisture': 0.25
        })
    return records