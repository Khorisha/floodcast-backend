import requests
import numpy as np
from datetime import datetime, timedelta

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
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        current = data.get('current', {})
        
        if not current:
            raise ValueError("No current weather data received")
        
        return {
            'temp': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'pressure': current.get('pressure_msl'),
            'wind_speed': current.get('wind_speed_10m'),
            'wind_dir': current.get('wind_direction_10m'),
            'rain_1h': current.get('precipitation', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Weather API error: {e}")
        raise Exception(f"Failed to fetch current weather: {str(e)}")

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
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get('hourly', {})
        
        if not hourly or not hourly.get('time'):
            raise ValueError("No forecast data received")
        
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
        raise Exception(f"Failed to fetch 7-day forecast: {str(e)}")

def get_historical_hours(hours_back=24):
    try:
        end = datetime.now()
        start = end - timedelta(hours=hours_back)
        
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude': PORT_LOUIS_LAT,
            'longitude': PORT_LOUIS_LON,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get('hourly', {})
        
        if not hourly or not hourly.get('time'):
            raise ValueError("No historical data received")
        
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
                'soil_moisture': hourly.get('soil_moisture_0_to_7cm', [0.25] * len(hourly['time']))[i] if 'soil_moisture_0_to_7cm' in hourly else 0.25
            })
        return records
    
    except Exception as e:
        print(f"Historical API error: {e}")
        raise Exception(f"Failed to fetch historical weather data: {str(e)}")

def get_historical_hours_for_date_range(start_time, end_time):
    try:
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude': PORT_LOUIS_LAT,
            'longitude': PORT_LOUIS_LON,
            'start_date': start_time.strftime('%Y-%m-%d'),
            'end_date': end_time.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get('hourly', {})
        
        if not hourly or not hourly.get('time'):
            raise ValueError("No historical data received for date range")
        
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
                'soil_moisture': hourly.get('soil_moisture_0_to_7cm', [0.25] * len(hourly['time']))[i] if 'soil_moisture_0_to_7cm' in hourly else 0.25
            })
        return records
    
    except Exception as e:
        print(f"Historical date range error: {e}")
        raise Exception(f"Failed to fetch historical data for date range: {str(e)}")