import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def plot_24hour_consumption(meter_id='AES2020896472402', target_date='2023-05-10'):
    """Plot 24-hour consumption from 12 AM to 12 AM next day"""
    
    print(f"Loading 24-hour data for meter {meter_id} on {target_date}...")
    
    # Load the combined data
    df = pd.read_csv('combined_load_profile_electrical.csv')
    df['Meter Datetime'] = pd.to_datetime(df['Meter Datetime'])
    df['Hour'] = df['Meter Datetime'].dt.hour
    df['Date'] = df['Meter Datetime'].dt.date
    
    # Convert target date to date object
    target_date_obj = pd.to_datetime(target_date).date()
    
    # Filter for specific meter and date
    meter_data = df[df['HES Meter Id'] == meter_id]
    day_data = meter_data[meter_data['Date'] == target_date_obj].copy()
    
    if len(day_data) == 0:
        print(f"No data found for meter {meter_id} on {target_date}")
        return None
    
    # Sort by time
    day_data = day_data.sort_values('Meter Datetime')
    
    print(f"Found {len(day_data)} records for {target_date}")
    
    # Create a clean 24-hour plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Define night hours for highlighting (9 PM to 4 AM)
    night_hours = list(range(21, 24)) + list(range(0, 5))
    
    # Create time axis starting from midnight
    start_time = pd.to_datetime(f"{target_date} 00:00:00")
    end_time = start_time + timedelta(days=1)
    
    # Plot the main consumption line
    ax.plot(day_data['Meter Datetime'], day_data['Import active power (QI+QIV)[W]'], 
             'b-', linewidth=2.5, alpha=0.8, label='Power Consumption')
    
    # Highlight nighttime periods with background shading
    for hour in night_hours:
        if hour >= 21:  # Evening night hours (21, 22, 23)
            night_start = pd.to_datetime(f"{target_date} {hour:02d}:00:00")
            night_end = pd.to_datetime(f"{target_date} {hour+1:02d}:00:00")
        else:  # Early morning night hours (0, 1, 2, 3, 4)
            next_day = (pd.to_datetime(target_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            night_start = pd.to_datetime(f"{next_day} {hour:02d}:00:00")
            night_end = pd.to_datetime(f"{next_day} {hour+1:02d}:00:00")
        
        ax.axvspan(night_start, night_end, alpha=0.2, color='red', label='Night Hours (9PM-4AM)' if hour == 21 else "")
    
    # Highlight nighttime data points
    night_data = day_data[day_data['Hour'].isin(night_hours)]
    if len(night_data) > 0:
        ax.scatter(night_data['Meter Datetime'], night_data['Import active power (QI+QIV)[W]'], 
                   color='red', alpha=0.8, s=40, zorder=5, label='Night Consumption Points')
    
    # Set title and labels
    ax.set_title(f'24-Hour Power Consumption - Meter {meter_id}\n{target_date} (12:00 AM to 12:00 AM next day)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time of Day', fontsize=14)
    ax.set_ylabel('Power Consumption (W)', fontsize=14)
    
    # Format x-axis to show hours from 0 to 24
    ax.set_xlim(start_time, end_time)
    
    # Set major ticks every 2 hours
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Set minor ticks every hour
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    # Customize the grid
    ax.grid(True, alpha=0.3, which='major', linestyle='-')
    ax.grid(True, alpha=0.1, which='minor', linestyle=':')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    # Add text box with key statistics
    consumption_values = day_data['Import active power (QI+QIV)[W]']
    night_consumption = night_data['Import active power (QI+QIV)[W]'] if len(night_data) > 0 else pd.Series([0])
    day_consumption = day_data[~day_data['Hour'].isin(night_hours)]['Import active power (QI+QIV)[W]']
    
    avg_consumption = consumption_values.mean()
    max_consumption = consumption_values.max()
    min_consumption = consumption_values.min()
    night_avg = night_consumption.mean() if len(night_consumption) > 0 else 0
    day_avg = day_consumption.mean() if len(day_consumption) > 0 else 0
    
    # Find peak time
    peak_time = day_data.loc[day_data['Import active power (QI+QIV)[W]'].idxmax(), 'Meter Datetime']
    
    stats_text = f"""Daily Statistics:
Average: {avg_consumption:.0f} W
Maximum: {max_consumption:.0f} W (at {peak_time.strftime('%H:%M')})
Minimum: {min_consumption:.0f} W
Day Avg: {day_avg:.0f} W
Night Avg: {night_avg:.0f} W
Night/Day Ratio: {night_avg/day_avg:.2f}"""
    
    # Add text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightblue', alpha=0.8), fontfamily='monospace')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Add horizontal lines for reference
    ax.axhline(y=avg_consumption, color='orange', linestyle='--', alpha=0.7, 
               linewidth=1.5, label=f'Daily Average ({avg_consumption:.0f}W)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = f'24hour_consumption_{meter_id}_{target_date.replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"24-hour consumption plot saved as: {filename}")
    
    # Print summary
    print(f"\n24-HOUR CONSUMPTION SUMMARY:")
    print(f"Date: {target_date} (12:00 AM to 12:00 AM next day)")
    print(f"Average consumption: {avg_consumption:.0f} W")
    print(f"Peak: {max_consumption:.0f} W at {peak_time.strftime('%H:%M')}")
    print(f"Minimum: {min_consumption:.0f} W")
    print(f"Day period average: {day_avg:.0f} W")
    print(f"Night period average: {night_avg:.0f} W")
    print(f"Night/Day ratio: {night_avg/day_avg:.2f}")
    
    if night_avg > day_avg:
        print("⚠️  ANOMALY: Night consumption exceeds day consumption!")
    
    return day_data

if __name__ == "__main__":
    # Generate the 24-hour consumption plot
    plot_24hour_consumption('AES2020896472402', '2023-05-10')