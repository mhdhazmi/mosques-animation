import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and process the data
df = pd.read_csv("cleaned_meter_KFM2020660190982.csv")
df["Meter Datetime"] = pd.to_datetime(df["Meter Datetime"])
df = df.sort_values("Meter Datetime")

# Add hour column for analysis
df['hour'] = df['Meter Datetime'].dt.hour
df['date'] = df['Meter Datetime'].dt.date

# Filter for late night hours (9 PM to 4 AM)
# This includes hours 21, 22, 23 (9 PM - 11:59 PM) and 0, 1, 2, 3 (12 AM - 3:59 AM)
night_hours = [21, 22, 23, 0, 1, 2, 3]
night_data = df[df['hour'].isin(night_hours)].copy()

# Group by date and calculate total consumption for night hours
daily_night_consumption = night_data.groupby('date')['Import active power (QI+QIV)[W]'].sum()

# Find the day with highest night consumption
max_consumption_date = daily_night_consumption.idxmax()
max_consumption_value = daily_night_consumption.max()

print(f"Day with highest night consumption (9 PM - 4 AM): {max_consumption_date}")
print(f"Total night consumption: {max_consumption_value:.2f} W")

# Get the full day's data for the identified date
full_day_data = df[df['date'] == max_consumption_date].copy()

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(full_day_data['hour'], full_day_data['Import active power (QI+QIV)[W]'], 
         marker='o', linewidth=3, markersize=6, color='blue')

# Highlight the night hours (9 PM - 4 AM)
night_full_day = full_day_data[full_day_data['hour'].isin(night_hours)]
plt.plot(night_full_day['hour'], night_full_day['Import active power (QI+QIV)[W]'], 
         marker='o', linewidth=4, markersize=8, color='red', label='Night Hours (9 PM - 4 AM)')

plt.title(f'Mosque Power Consumption - {max_consumption_date}\n(Highest Night Consumption Day)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Power Consumption (W)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set x-axis to show all hours
plt.xticks(range(0, 24, 2))
plt.xlim(-0.5, 23.5)

# Add some statistics as text
avg_night = night_full_day['Import active power (QI+QIV)[W]'].mean()
max_night = night_full_day['Import active power (QI+QIV)[W]'].max()
min_night = night_full_day['Import active power (QI+QIV)[W]'].min()

stats_text = f'Night Hours Stats:\nAvg: {avg_night:.1f}W\nMax: {max_night:.1f}W\nMin: {min_night:.1f}W'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('highest_night_consumption_day.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print(f"\nDetailed Analysis for {max_consumption_date}:")
print("=" * 50)
for _, row in full_day_data.iterrows():
    hour = row['hour']
    power = row['Import active power (QI+QIV)[W]']
    if hour in night_hours:
        print(f"Hour {hour:2d}: {power:6.1f}W  *** NIGHT HOUR ***")
    else:
        print(f"Hour {hour:2d}: {power:6.1f}W")

print(f"\nNight consumption summary:")
print(f"Total night consumption: {night_full_day['Import active power (QI+QIV)[W]'].sum():.1f}W")
print(f"Average night consumption: {avg_night:.1f}W")
print(f"Peak night consumption: {max_night:.1f}W at hour {night_full_day.loc[night_full_day['Import active power (QI+QIV)[W]'].idxmax(), 'hour']}")