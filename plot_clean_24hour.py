import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def plot_clean_24hour_consumption(meter_id='AES2020896472402', target_date='2023-05-10'):
    """Plot clean 24-hour consumption with seaborn styling"""
    
    print(f"Loading 24-hour data for meter {meter_id} on {target_date}...")
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
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
    
    # Create a clean plot with seaborn styling
    plt.figure(figsize=(14, 8))
    
    # Create time axis starting from midnight
    start_time = pd.to_datetime(f"{target_date} 00:00:00")
    end_time = start_time + timedelta(days=1)
    
    # Plot the main consumption line with seaborn color
    plt.plot(day_data['Meter Datetime'], day_data['Import active power (QI+QIV)[W]'], 
             linewidth=2.5, alpha=0.9, color=sns.color_palette("husl", 8)[0])
    
    # Set title and labels with clean styling
    plt.title(f'24-Hour Power Consumption - Meter {meter_id}\n{target_date}', 
              fontsize=16, fontweight='normal', pad=20, color='#2E2E2E')
    plt.xlabel('Time of Day', fontsize=13, color='#2E2E2E')
    plt.ylabel('Power Consumption (W)', fontsize=13, color='#2E2E2E')
    
    # Format x-axis to show hours from 0 to 24
    plt.xlim(start_time, end_time)
    
    # Set major ticks every 2 hours
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Set minor ticks every hour
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    # Customize the grid with seaborn style
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Style the axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Color the tick labels
    ax.tick_params(colors='#2E2E2E', which='both')
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    # Adjust layout with seaborn style margins
    plt.tight_layout()
    
    # Save the plot
    filename = f'clean_24hour_consumption_{meter_id}_{target_date.replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Clean 24-hour consumption plot saved as: {filename}")
    
    # Print basic summary
    consumption_values = day_data['Import active power (QI+QIV)[W]']
    avg_consumption = consumption_values.mean()
    max_consumption = consumption_values.max()
    min_consumption = consumption_values.min()
    
    print(f"\nConsumption Summary for {target_date}:")
    print(f"Average: {avg_consumption:.0f} W")
    print(f"Maximum: {max_consumption:.0f} W")
    print(f"Minimum: {min_consumption:.0f} W")
    
    return day_data

if __name__ == "__main__":
    # Generate the clean 24-hour consumption plot
    plot_clean_24hour_consumption('AES2020896472402', '2023-05-10')