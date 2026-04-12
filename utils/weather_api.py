import requests
import csv
import os
import math
from datetime import datetime, timedelta

PORT_LOUIS_LAT = -20.1609
PORT_LOUIS_LON = 57.5012

ERA5_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'era5_raw.csv')
ERA5_START = datetime(2010, 1, 1)
ERA5_END   = datetime(2024, 12, 31, 23, 0, 0)

# Rainfall stored RAW — predictor applies convective correction internally.
# Soil moisture is scaled here because the predictor does not adjust it.
_era5_cache = None


def _load_era5_cache():
    global _era5_cache
    if _era5_cache is not None:
        return _era5_cache
    try:
        print("Loading ERA5 CSV into memory…")
        cache = {}
        with open(ERA5_CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts  = row['Timestamp']
                key = ts[:10] + 'T' + ts[11:16]
                cache[key] = {
                    'time':               key,
                    'temperature':        float(row['Temperature_C']),
                    'humidity':           float(row['Humidity_pct']),
                    'rainfall':           float(row['Rainfall_raw_mmhr']),
                    'pressure':           float(row['Pressure_hPa']),
                    'wind_speed':         float(row['WindSpeed_ms']),
                    'wind_dir':           float(row['WindDir_deg']),
                    'soil_moisture':      float(row['SoilMoist_top_m3']),
                    'soil_moisture_deep': float(row['SoilMoist_deep_m3']),
                }
        _era5_cache = cache
        print(f"ERA5 CSV loaded: {len(cache)} records")
    except Exception as e:
        print(f"WARNING: could not load ERA5 CSV: {e}")
        _era5_cache = {}
    return _era5_cache


def _v(value, default):
    return value if value is not None else default


def _scale_soil_moisture_to_era5(sm):
    """Scale Open-Meteo forecast SM (0.15-0.25) to ERA5 range (0.22-0.35)."""
    if sm is None:
        return 0.25
    scaled = 0.20 + (float(sm) - 0.15) * 1.2
    return max(0.20, min(0.40, scaled))


def _transform_forecast_record(raw):
    """
    Adapt a raw Open-Meteo record to match ERA5 distribution.
    Rainfall is passed through RAW so the predictor can apply its convective correction.
    Soil moisture is scaled to ERA5 range.
    """
    soil_top  = _scale_soil_moisture_to_era5(raw.get('soil_moisture', 0.25))
    soil_deep = soil_top * 0.9
    return {
        'time':               raw.get('time', ''),
        'temperature':        float(raw.get('temperature', 25) or 25),
        'humidity':           float(raw.get('humidity', 70) or 70),
        'rainfall':           float(raw.get('rainfall', 0) or 0),
        'pressure':           float(raw.get('pressure', 1013) or 1013),
        'wind_speed':         float(raw.get('wind_speed', 5) or 5),
        'wind_dir':           float(raw.get('wind_dir', 180) or 180),
        'soil_moisture':      soil_top,
        'soil_moisture_deep': soil_deep,
    }


def _era5_available(from_dt, to_dt):
    cache = _load_era5_cache()
    if not cache:
        return False
    return from_dt >= ERA5_START and to_dt <= ERA5_END + timedelta(hours=23)


def get_era5_hours_for_range(from_dt, to_dt):
    cache   = _load_era5_cache()
    records = []
    cur     = from_dt.replace(minute=0, second=0, microsecond=0)
    end     = to_dt.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        key = cur.strftime('%Y-%m-%dT%H:%M')
        rec = cache.get(key)
        records.append(rec if rec else {
            'time': key, 'temperature': 25.0, 'humidity': 70.0,
            'rainfall': 0.0, 'pressure': 1013.0, 'wind_speed': 5.0,
            'wind_dir': 180.0, 'soil_moisture': 0.25, 'soil_moisture_deep': 0.30,
        })
        cur += timedelta(hours=1)
    return records


def get_current_weather():
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': PORT_LOUIS_LAT, 'longitude': PORT_LOUIS_LON,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        current = response.json().get('current', {})
        if not current:
            raise ValueError("No current weather data received")
        return {
            'temp':       current.get('temperature_2m'),
            'humidity':   current.get('relative_humidity_2m'),
            'pressure':   current.get('pressure_msl'),
            'wind_speed': current.get('wind_speed_10m'),
            'wind_dir':   current.get('wind_direction_10m'),
            'rain_1h':    current.get('precipitation', 0),
            'timestamp':  datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        raise Exception(f"Failed to fetch current weather: {str(e)}")


def get_forecast_7days():
    """7-day hourly forecast with soil moisture scaled to ERA5 range."""
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': PORT_LOUIS_LAT, 'longitude': PORT_LOUIS_LON,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
            'forecast_days': 8, 'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        hourly = response.json().get('hourly', {})
        if not hourly or not hourly.get('time'):
            raise ValueError("No forecast data received")
        sm_list = hourly.get('soil_moisture_0_to_7cm', [])
        records = []
        for i in range(len(hourly['time'])):
            records.append(_transform_forecast_record({
                'time':         hourly['time'][i],
                'temperature':  _v(hourly['temperature_2m'][i], 25.0),
                'humidity':     _v(hourly['relative_humidity_2m'][i], 70.0),
                'rainfall':     _v(hourly['precipitation'][i], 0.0),
                'pressure':     _v(hourly['pressure_msl'][i], 1013.0),
                'wind_speed':   _v(hourly['wind_speed_10m'][i], 5.0),
                'wind_dir':     _v(hourly['wind_direction_10m'][i], 180.0),
                'soil_moisture': _v(sm_list[i] if sm_list else None, 0.25),
            }))
        return records
    except Exception as e:
        print(f"Forecast API error: {e}")
        raise Exception(f"Failed to fetch 7-day forecast: {str(e)}")


def get_historical_hours(hours_back=168):
    """
    Most recent hours_back hours.
    Uses ERA5 CSV for 2010-2024; forecast API with SM scaling for 2025+.
    """
    now     = datetime.now()
    from_dt = now - timedelta(hours=hours_back)
    if _era5_available(from_dt, now):
        return get_era5_hours_for_range(from_dt, now)
    try:
        past_days = max(2, math.ceil(hours_back / 24) + 2)
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': PORT_LOUIS_LAT, 'longitude': PORT_LOUIS_LON,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
            'past_days': past_days, 'forecast_days': 1, 'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        hourly = response.json().get('hourly', {})
        if not hourly or not hourly.get('time'):
            raise ValueError("No historical data received")
        sm_list = hourly.get('soil_moisture_0_to_7cm', [])
        records = []
        for i in range(len(hourly['time'])):
            records.append(_transform_forecast_record({
                'time':         hourly['time'][i],
                'temperature':  _v(hourly['temperature_2m'][i], 25.0),
                'humidity':     _v(hourly['relative_humidity_2m'][i], 70.0),
                'rainfall':     _v(hourly['precipitation'][i], 0.0),
                'pressure':     _v(hourly['pressure_msl'][i], 1013.0),
                'wind_speed':   _v(hourly['wind_speed_10m'][i], 5.0),
                'wind_dir':     _v(hourly['wind_direction_10m'][i], 180.0),
                'soil_moisture': _v(sm_list[i] if sm_list else None, 0.25),
            }))
        return records[-hours_back:] if len(records) >= hours_back else records
    except Exception as e:
        print(f"Historical API error: {e}")
        raise Exception(f"Failed to fetch historical weather data: {str(e)}")


def get_hours_for_week(from_date, num_days=7):
    """
    num_days consecutive days from from_date.
    Priority: ERA5 CSV → forecast API → archive API.
    """
    end_date = from_date + timedelta(days=num_days - 1)
    if _era5_available(from_date, end_date):
        return get_era5_hours_for_range(
            from_date.replace(hour=0, minute=0, second=0),
            end_date.replace(hour=23, minute=0, second=0)
        )
    today     = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    from_diff = (from_date.date() - today.date()).days
    try:
        if from_diff >= -8:
            past_days     = max(0, -from_diff) + 1
            forecast_days = max(1, (end_date.date() - today.date()).days + 2)
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': PORT_LOUIS_LAT, 'longitude': PORT_LOUIS_LON,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
                'past_days': past_days, 'forecast_days': forecast_days, 'timezone': 'Indian/Mauritius'
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            hourly = response.json().get('hourly', {})
            if not hourly or not hourly.get('time'):
                raise ValueError("No data from forecast API")
            from_str = from_date.strftime('%Y-%m-%d')
            end_str  = end_date.strftime('%Y-%m-%d')
            sm_list  = hourly.get('soil_moisture_0_to_7cm', [])
            records  = []
            for i in range(len(hourly['time'])):
                t = hourly['time'][i]
                if from_str <= t[:10] <= end_str:
                    records.append(_transform_forecast_record({
                        'time':         t,
                        'temperature':  _v(hourly['temperature_2m'][i], 25.0),
                        'humidity':     _v(hourly['relative_humidity_2m'][i], 70.0),
                        'rainfall':     _v(hourly['precipitation'][i], 0.0),
                        'pressure':     _v(hourly['pressure_msl'][i], 1013.0),
                        'wind_speed':   _v(hourly['wind_speed_10m'][i], 5.0),
                        'wind_dir':     _v(hourly['wind_direction_10m'][i], 180.0),
                        'soil_moisture': _v(sm_list[i] if sm_list else None, 0.25),
                    }))
            return records
        else:
            return _get_archive_hours(
                from_date.replace(hour=0, minute=0, second=0),
                end_date.replace(hour=23, minute=0, second=0)
            )
    except Exception as e:
        print(f"get_hours_for_week error: {e}")
        raise Exception(f"Failed to fetch week of data: {str(e)}")


def _get_archive_hours(start_time, end_time):
    """ERA5 archive API for 2025+ dates older than 8 days."""
    try:
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude': PORT_LOUIS_LAT, 'longitude': PORT_LOUIS_LON,
            'start_date': start_time.strftime('%Y-%m-%d'),
            'end_date':   end_time.strftime('%Y-%m-%d'),
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,soil_moisture_0_to_7cm',
            'timezone': 'Indian/Mauritius'
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        hourly = response.json().get('hourly', {})
        if not hourly or not hourly.get('time'):
            raise ValueError("No archive data received")
        sm_list = hourly.get('soil_moisture_0_to_7cm', [])
        records = []
        for i in range(len(hourly['time'])):
            records.append(_transform_forecast_record({
                'time':         hourly['time'][i],
                'temperature':  _v(hourly['temperature_2m'][i], 25.0),
                'humidity':     _v(hourly['relative_humidity_2m'][i], 70.0),
                'rainfall':     _v(hourly['precipitation'][i], 0.0),
                'pressure':     _v(hourly['pressure_msl'][i], 1013.0),
                'wind_speed':   _v(hourly['wind_speed_10m'][i], 5.0),
                'wind_dir':     _v(hourly['wind_direction_10m'][i], 180.0),
                'soil_moisture': _v(sm_list[i] if sm_list else None, 0.25),
            }))
        return records
    except Exception as e:
        print(f"Archive API error: {e}")
        raise Exception(f"Failed to fetch archive data: {str(e)}")


def get_historical_hours_for_date_range(start_time, end_time):
    if _era5_available(start_time, end_time):
        return get_era5_hours_for_range(start_time, end_time)
    return _get_archive_hours(start_time, end_time)
