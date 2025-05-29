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

# Filter for morning hours (6 AM to 11 AM)
# This includes hours 6, 7, 8, 9, 10
morning_hours = [6, 7, 8, 9, 10]
morning_data = df[df['hour'].isin(morning_hours)].copy()

# Group by date and calculate total consumption for morning hours
daily_morning_consumption = morning_data.groupby('date')['Import active power (QI+QIV)[W]'].sum()

# Find the day with highest morning consumption
max_consumption_date = daily_morning_consumption.idxmax()
max_consumption_value = daily_morning_consumption.max()

print(f"Day with highest morning consumption (6 AM - 11 AM): {max_consumption_date}")
print(f"Total morning consumption: {max_consumption_value:.2f} W")

# Get the full day's data for the identified date
full_day_data = df[df['date'] == max_consumption_date].copy()

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(full_day_data['hour'], full_day_data['Import active power (QI+QIV)[W]'], 
         marker='o', linewidth=3, markersize=6, color='blue')

# Highlight the morning hours (6 AM - 11 AM)
morning_full_day = full_day_data[full_day_data['hour'].isin(morning_hours)]
plt.plot(morning_full_day['hour'], morning_full_day['Import active power (QI+QIV)[W]'], 
         marker='o', linewidth=4, markersize=8, color='orange', label='Morning Hours (6 AM - 11 AM)')

plt.title(f'Mosque Power Consumption - {max_consumption_date}\n(Highest Morning Consumption Day)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Power Consumption (W)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Set x-axis to show all hours
plt.xticks(range(0, 24, 2))
plt.xlim(-0.5, 23.5)

# Add some statistics as text
avg_morning = morning_full_day['Import active power (QI+QIV)[W]'].mean()
max_morning = morning_full_day['Import active power (QI+QIV)[W]'].max()
min_morning = morning_full_day['Import active power (QI+QIV)[W]'].min()

stats_text = f'Morning Hours Stats:\nAvg: {avg_morning:.1f}W\nMax: {max_morning:.1f}W\nMin: {min_morning:.1f}W'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('highest_morning_consumption_day.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print(f"\nDetailed Analysis for {max_consumption_date}:")
print("=" * 50)
for _, row in full_day_data.iterrows():
    hour = row['hour']
    power = row['Import active power (QI+QIV)[W]']
    if hour in morning_hours:
        print(f"Hour {hour:2d}: {power:6.1f}W  *** MORNING HOUR ***")
    else:
        print(f"Hour {hour:2d}: {power:6.1f}W")

print(f"\nMorning consumption summary:")
print(f"Total morning consumption: {morning_full_day['Import active power (QI+QIV)[W]'].sum():.1f}W")
print(f"Average morning consumption: {avg_morning:.1f}W")
print(f"Peak morning consumption: {max_morning:.1f}W at hour {morning_full_day.loc[morning_full_day['Import active power (QI+QIV)[W]'].idxmax(), 'hour']}")

# Also show top 5 days for comparison
print(f"\nTop 5 days with highest morning consumption:")
print("=" * 50)
top_5_days = daily_morning_consumption.nlargest(5)
for i, (date, consumption) in enumerate(top_5_days.items(), 1):
    print(f"{i}. {date}: {consumption:.1f}W")