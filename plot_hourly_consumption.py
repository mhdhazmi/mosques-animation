import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from datetime import datetime
import seaborn as sns

# Set style for professional appearance
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")

# Read the CSV file
df = pd.read_csv('cleaned_meter_KFM2020660190982.csv')

# Convert datetime column to datetime type
df['Meter Datetime'] = pd.to_datetime(df['Meter Datetime'])

# Extract hour and date
df['Hour'] = df['Meter Datetime'].dt.hour + df['Meter Datetime'].dt.minute/60
df['Date'] = df['Meter Datetime'].dt.date

# Group by date and hour, taking mean of power consumption for each hour
hourly_data = df.groupby(['Date', 'Hour'])['Import active power (QI+QIV)[W]'].mean().reset_index()

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique dates
unique_dates = hourly_data['Date'].unique()

# Sample a subset of dates for better visualization (every 7th day to avoid overcrowding)
sample_dates = unique_dates[::7]

# Plot each day with transparency and smoothing
for i, date in enumerate(sample_dates):
    day_data = hourly_data[hourly_data['Date'] == date]
    
    if len(day_data) > 5:  # Only plot if we have sufficient data points
        hours = day_data['Hour'].values
        power = day_data['Import active power (QI+QIV)[W]'].values
        
        # Create smooth curve using interpolation
        if len(hours) > 3:
            # Sort by hour to ensure proper interpolation
            sorted_indices = np.argsort(hours)
            hours_sorted = hours[sorted_indices]
            power_sorted = power[sorted_indices]
            
            # Create interpolation function
            f = interpolate.interp1d(hours_sorted, power_sorted, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            
            # Generate smooth curve
            hours_smooth = np.linspace(0, 23.5, 100)
            power_smooth = f(hours_smooth)
            
            # Plot the smooth curve
            ax.plot(hours_smooth, power_smooth, color='gray', alpha=0.3, linewidth=1.5)

# Calculate and plot the average daily pattern
avg_hourly = hourly_data.groupby('Hour')['Import active power (QI+QIV)[W]'].mean()
hours_avg = avg_hourly.index.values
power_avg = avg_hourly.values

# Smooth the average curve
f_avg = interpolate.interp1d(hours_avg, power_avg, kind='cubic', 
                           bounds_error=False, fill_value='extrapolate')
hours_avg_smooth = np.linspace(0, 23.5, 100)
power_avg_smooth = f_avg(hours_avg_smooth)

# Plot average pattern with emphasis
ax.plot(hours_avg_smooth, power_avg_smooth, color='red', linewidth=3, alpha=0.8)

# Customize the plot
ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Power Consumption', fontsize=12, fontweight='bold')

# Set x-axis ticks to show hours
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])

ax.set_xlim(0, 24)

# Adjust layout and save
plt.tight_layout()
plt.savefig('hourly_consumption_pattern.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'hourly_consumption_pattern.png'")
print(f"Processed {len(unique_dates)} days of data from {len(sample_dates)} sampled days shown")