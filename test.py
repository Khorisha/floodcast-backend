# Create test.py in your backend folder
import pandas as pd

df = pd.read_csv('data/era5_raw.csv', parse_dates=['Timestamp'])

# Find days with heavy rain in 2023-2024
df['Rainfall_corrected'] = df['Rainfall_raw_mmhr'].apply(lambda x: x * 1.6 if x >= 2 else x)

# Get top 10 rainiest days in test period (2023-2024)
test_data = df[(df['Timestamp'] >= '2023-01-01') & (df['Timestamp'] <= '2024-12-31')]
daily_rain = test_data.groupby(test_data['Timestamp'].dt.date)['Rainfall_corrected'].sum().sort_values(ascending=False)

print("Top 10 rainiest days in test set:")
for date, rain in daily_rain.head(10).items():
    print(f"  {date}: {rain:.1f}mm total rain")