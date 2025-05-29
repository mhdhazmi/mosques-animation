import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

def plot_meter_specific_day(meter_id='AES2020896472402', target_date='2023-05-10'):
    """Plot consumption for a specific meter on a specific day"""
    
    print(f"Loading data for meter {meter_id} on {target_date}...")
    
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
        # Let's check what dates are available for this meter
        print("Available dates for this meter:")
        available_dates = meter_data['Date'].unique()
        available_dates = sorted(available_dates)
        print(f"Date range: {available_dates[0]} to {available_dates[-1]}")
        print(f"Total available dates: {len(available_dates)}")
        
        # Show some dates around the target date
        target_date_pd = pd.to_datetime(target_date).date()
        nearby_dates = [d for d in available_dates if abs((d - target_date_pd).days) <= 7]
        if nearby_dates:
            print(f"Dates within 7 days of {target_date}: {nearby_dates}")
            # Use the closest available date
            closest_date = min(nearby_dates, key=lambda x: abs((x - target_date_pd).days))
            print(f"Using closest available date: {closest_date}")
            day_data = meter_data[meter_data['Date'] == closest_date].copy()
            target_date = str(closest_date)
        else:
            return None
    
    if len(day_data) == 0:
        print("Still no data found")
        return None
    
    # Sort by time
    day_data = day_data.sort_values('Meter Datetime')
    
    print(f"Found {len(day_data)} records for {target_date}")
    print(f"Time range: {day_data['Meter Datetime'].min()} to {day_data['Meter Datetime'].max()}")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define night hours for highlighting
    night_hours = list(range(21, 24)) + list(range(0, 5))
    
    # Plot 1: Full day time series
    ax1 = axes[0, 0]
    ax1.plot(day_data['Meter Datetime'], day_data['Import active power (QI+QIV)[W]'], 
             'b-', linewidth=2, alpha=0.8)
    
    # Highlight nighttime periods
    night_data = day_data[day_data['Hour'].isin(night_hours)]
    if len(night_data) > 0:
        ax1.scatter(night_data['Meter Datetime'], night_data['Import active power (QI+QIV)[W]'], 
                   color='red', alpha=0.7, s=30, label='Night hours (9PM-4AM)', zorder=5)
    
    ax1.set_title(f'Meter {meter_id}\nFull Day Consumption - {target_date}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (W)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis to show hours
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Hourly aggregation
    ax2 = axes[0, 1]
    if len(day_data) > 0:
        hourly_data = day_data.groupby('Hour')['Import active power (QI+QIV)[W]'].agg(['mean', 'max', 'min'])
        
        # Create color array for bars
        colors = ['red' if hour in night_hours else 'blue' for hour in hourly_data.index]
        
        bars = ax2.bar(hourly_data.index, hourly_data['mean'], color=colors, alpha=0.7, label='Average')
        ax2.plot(hourly_data.index, hourly_data['max'], 'r-', marker='o', markersize=4, label='Max')
        ax2.plot(hourly_data.index, hourly_data['min'], 'g-', marker='s', markersize=4, label='Min')
        
        ax2.set_title('Hourly Statistics')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Power (W)')
        ax2.set_xticks(range(0, 24, 2))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Consumption distribution
    ax3 = axes[1, 0]
    consumption_values = day_data['Import active power (QI+QIV)[W]']
    
    ax3.hist(consumption_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(consumption_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {consumption_values.mean():.0f}W')
    ax3.axvline(consumption_values.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {consumption_values.median():.0f}W')
    
    ax3.set_title('Consumption Distribution')
    ax3.set_xlabel('Power (W)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics and anomaly indicators
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate detailed statistics
    total_consumption = consumption_values.sum()
    avg_consumption = consumption_values.mean()
    max_consumption = consumption_values.max()
    min_consumption = consumption_values.min()
    
    # Night vs day stats
    night_consumption = night_data['Import active power (QI+QIV)[W]'] if len(night_data) > 0 else pd.Series([0])
    day_consumption = day_data[~day_data['Hour'].isin(night_hours)]['Import active power (QI+QIV)[W]']
    
    night_avg = night_consumption.mean() if len(night_consumption) > 0 else 0
    day_avg = day_consumption.mean() if len(day_consumption) > 0 else 0
    
    # Find peak consumption time
    peak_time = day_data.loc[day_data['Import active power (QI+QIV)[W]'].idxmax(), 'Meter Datetime'] if len(day_data) > 0 else None
    peak_hour = peak_time.hour if peak_time else None
    
    # Check for anomalies
    anomaly_indicators = []
    if night_avg > day_avg:
        anomaly_indicators.append("⚠️ Night consumption > Day consumption")
    if peak_hour and peak_hour in night_hours:
        anomaly_indicators.append("⚠️ Peak consumption during night hours")
    if min_consumption > avg_consumption * 0.5:
        anomaly_indicators.append("⚠️ High baseline consumption")
    
    stats_text = f"""
DAILY CONSUMPTION ANALYSIS
{target_date}

Basic Statistics:
• Average: {avg_consumption:.0f} W
• Maximum: {max_consumption:.0f} W  
• Minimum: {min_consumption:.0f} W
• Total Energy: {total_consumption/1000:.1f} kWh

Time-based Analysis:
• Day average: {day_avg:.0f} W
• Night average: {night_avg:.0f} W
• Night/Day ratio: {night_avg/day_avg:.2f}
• Peak time: {peak_time.strftime('%H:%M') if peak_time else 'N/A'}

Data Coverage:
• Records: {len(day_data)}
• Time span: {(day_data['Meter Datetime'].max() - day_data['Meter Datetime'].min()).total_seconds()/3600:.1f} hours

ANOMALY INDICATORS:
{chr(10).join(anomaly_indicators) if anomaly_indicators else "✅ No major anomalies detected"}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Detailed Daily Analysis - Meter {meter_id} - {target_date}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    filename = f'meter_{meter_id}_day_{target_date.replace("-", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as: {filename}")
    
    # Print key findings
    print(f"\nKEY FINDINGS for {target_date}:")
    print(f"• Average consumption: {avg_consumption:.0f}W")
    print(f"• Peak consumption: {max_consumption:.0f}W at {peak_time.strftime('%H:%M') if peak_time else 'N/A'}")
    print(f"• Night vs Day consumption: {night_avg:.0f}W vs {day_avg:.0f}W (ratio: {night_avg/day_avg:.2f})")
    
    if anomaly_indicators:
        print(f"• Anomalies detected: {len(anomaly_indicators)}")
        for anomaly in anomaly_indicators:
            print(f"  {anomaly}")
    
    return day_data

def check_available_dates(meter_id='AES2020896472402'):
    """Check what dates are available for the meter around the target date"""
    
    print(f"Checking available dates for meter {meter_id}...")
    
    df = pd.read_csv('combined_load_profile_electrical.csv')
    df['Meter Datetime'] = pd.to_datetime(df['Meter Datetime'])
    df['Date'] = df['Meter Datetime'].dt.date
    
    meter_data = df[df['HES Meter Id'] == meter_id]
    
    if len(meter_data) == 0:
        print(f"No data found for meter {meter_id}")
        return
    
    available_dates = sorted(meter_data['Date'].unique())
    print(f"Available date range: {available_dates[0]} to {available_dates[-1]}")
    print(f"Total available dates: {len(available_dates)}")
    
    # Show dates in May 2023
    may_2023_dates = [d for d in available_dates if d.year == 2023 and d.month == 5]
    if may_2023_dates:
        print(f"Available dates in May 2023: {may_2023_dates}")
    else:
        print("No data available in May 2023")
        
    return available_dates

if __name__ == "__main__":
    # First check available dates
    available_dates = check_available_dates('AES2020896472402')
    
    print("\n" + "="*50)
    
    # Try to plot the specific date
    day_data = plot_meter_specific_day('AES2020896472402', '2023-05-10')