import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_specific_meter(meter_id='AES2020896472402'):
    """Detailed analysis of the most anomalous meter"""
    
    print(f"Detailed analysis of meter: {meter_id}")
    
    # Load data for specific meter
    print("Loading data...")
    df = pd.read_csv('combined_load_profile_electrical.csv')
    df['Meter Datetime'] = pd.to_datetime(df['Meter Datetime'])
    df['Hour'] = df['Meter Datetime'].dt.hour
    df['Date'] = df['Meter Datetime'].dt.date
    df['Day_of_week'] = df['Meter Datetime'].dt.day_name()
    
    # Filter for specific meter
    meter_data = df[df['HES Meter Id'] == meter_id].copy()
    
    if len(meter_data) == 0:
        print(f"No data found for meter {meter_id}")
        return
    
    print(f"Records for meter {meter_id}: {len(meter_data):,}")
    print(f"Date range: {meter_data['Meter Datetime'].min()} to {meter_data['Meter Datetime'].max()}")
    
    # Create comprehensive analysis plots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: 24-hour consumption pattern
    ax1 = plt.subplot(3, 3, 1)
    hourly_stats = meter_data.groupby('Hour')['Import active power (QI+QIV)[W]'].agg(['mean', 'median', 'max', 'min'])
    
    night_hours = list(range(21, 24)) + list(range(0, 5))
    colors = ['red' if hour in night_hours else 'blue' for hour in hourly_stats.index]
    
    bars = ax1.bar(hourly_stats.index, hourly_stats['mean'], color=colors, alpha=0.7, label='Average')
    ax1.plot(hourly_stats.index, hourly_stats['max'], 'r-', marker='o', markersize=3, label='Max', alpha=0.8)
    ax1.plot(hourly_stats.index, hourly_stats['min'], 'g-', marker='s', markersize=3, label='Min', alpha=0.8)
    
    ax1.set_title(f'Meter {meter_id}: Hourly Consumption Patterns')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Power (W)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # Plot 2: Weekly pattern
    ax2 = plt.subplot(3, 3, 2)
    weekly_pattern = meter_data.groupby('Day_of_week')['Import active power (QI+QIV)[W]'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern = weekly_pattern.reindex(day_order)
    
    ax2.bar(range(len(weekly_pattern)), weekly_pattern.values, color='orange', alpha=0.7)
    ax2.set_title('Average Consumption by Day of Week')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Power (W)')
    ax2.set_xticks(range(len(day_order)))
    ax2.set_xticklabels([day[:3] for day in day_order], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Monthly trend
    ax3 = plt.subplot(3, 3, 3)
    meter_data['YearMonth'] = meter_data['Meter Datetime'].dt.to_period('M')
    monthly_trend = meter_data.groupby('YearMonth')['Import active power (QI+QIV)[W]'].mean()
    
    ax3.plot(range(len(monthly_trend)), monthly_trend.values, 'o-', color='purple', linewidth=2, markersize=6)
    ax3.set_title('Monthly Consumption Trend')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Power (W)')
    ax3.set_xticks(range(len(monthly_trend)))
    ax3.set_xticklabels([str(period) for period in monthly_trend.index], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Night vs Day comparison
    ax4 = plt.subplot(3, 3, 4)
    night_data = meter_data[meter_data['Hour'].isin(night_hours)]['Import active power (QI+QIV)[W]']
    day_data = meter_data[~meter_data['Hour'].isin(night_hours)]['Import active power (QI+QIV)[W]']
    
    ax4.hist([night_data, day_data], bins=50, alpha=0.7, label=['Night (9PM-4AM)', 'Day'], color=['red', 'blue'])
    ax4.set_title('Distribution: Night vs Day Consumption')
    ax4.set_xlabel('Power (W)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Time series sample (1 month)
    ax5 = plt.subplot(3, 3, (5, 6))
    sample_data = meter_data[meter_data['Meter Datetime'].dt.month == 8].sort_values('Meter Datetime')  # August data
    if len(sample_data) > 0:
        ax5.plot(sample_data['Meter Datetime'], sample_data['Import active power (QI+QIV)[W]'], 
                alpha=0.8, linewidth=0.8, color='darkblue')
        
        # Highlight nighttime points
        night_mask = sample_data['Hour'].isin(night_hours)
        night_sample = sample_data[night_mask]
        ax5.scatter(night_sample['Meter Datetime'], night_sample['Import active power (QI+QIV)[W]'], 
                   color='red', alpha=0.6, s=8, label='Night consumption')
        
        ax5.set_title('Sample Time Series (August 2022)')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Power (W)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 6: Consumption heatmap by hour and day
    ax6 = plt.subplot(3, 3, (7, 8))
    meter_data['Date_only'] = meter_data['Meter Datetime'].dt.date
    
    # Create pivot table for heatmap (sample recent data)
    recent_data = meter_data[meter_data['Meter Datetime'] >= meter_data['Meter Datetime'].max() - pd.Timedelta(days=30)]
    if len(recent_data) > 0:
        pivot_data = recent_data.pivot_table(
            values='Import active power (QI+QIV)[W]', 
            index='Date_only', 
            columns='Hour', 
            aggfunc='mean'
        )
        
        # Select last 15 days for visibility
        if len(pivot_data) > 15:
            pivot_data = pivot_data.tail(15)
        
        sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Power (W)'}, ax=ax6)
        ax6.set_title('Recent Daily Consumption Pattern (Last 30 days)')
        ax6.set_xlabel('Hour of Day')
        ax6.set_ylabel('Date')
        
        # Highlight night hours
        for hour in night_hours:
            if hour in pivot_data.columns:
                ax6.axvline(x=pivot_data.columns.get_loc(hour)+0.5, color='blue', linestyle='--', alpha=0.7)
    
    # Plot 7: Statistics summary
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')
    
    # Calculate key statistics
    overall_mean = meter_data['Import active power (QI+QIV)[W]'].mean()
    overall_max = meter_data['Import active power (QI+QIV)[W]'].max()
    overall_min = meter_data['Import active power (QI+QIV)[W]'].min()
    night_mean = night_data.mean()
    day_mean = day_data.mean()
    night_min = night_data.min()
    night_max = night_data.max()
    
    stats_text = f"""
    ANOMALY ANALYSIS SUMMARY
    
    Overall Statistics:
    • Mean consumption: {overall_mean:.0f} W
    • Maximum: {overall_max:.0f} W
    • Minimum: {overall_min:.0f} W
    
    Day vs Night Comparison:
    • Day average: {day_mean:.0f} W
    • Night average: {night_mean:.0f} W
    • Night/Day ratio: {night_mean/day_mean:.2f}
    
    Night Consumption (9PM-4AM):
    • Minimum: {night_min:.0f} W
    • Maximum: {night_max:.0f} W
    • Base load ratio: {night_min/overall_mean:.2f}
    
    ANOMALY INDICATORS:
    ✓ Night avg > Day avg: {night_mean > day_mean}
    ✓ High night minimum: {night_min > 0.5 * overall_mean}
    ✓ Night peak > 70% overall max: {night_max > 0.7 * overall_max}
    """
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Comprehensive Anomaly Analysis - Meter {meter_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'detailed_analysis_{meter_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed analysis saved to: detailed_analysis_{meter_id}.png")
    
    # Print key findings
    print(f"\nKEY FINDINGS for Meter {meter_id}:")
    print(f"• Night consumption ({night_mean:.0f}W) is {night_mean/day_mean:.1f}x higher than day ({day_mean:.0f}W)")
    print(f"• Night minimum consumption ({night_min:.0f}W) is {night_min/overall_mean:.1f}x the overall average")
    print(f"• This indicates potential: inappropriate usage, faulty equipment, or irregular consumption patterns")
    
    return meter_data

if __name__ == "__main__":
    # Analyze the most anomalous meter
    meter_data = analyze_specific_meter('AES2020896472402')
    
    # Also create a simple comparison plot for multiple anomalous meters
    print("\nCreating comparison plot for top anomalous meters...")
    
    df = pd.read_csv('combined_load_profile_electrical.csv')
    df['Meter Datetime'] = pd.to_datetime(df['Meter Datetime'])
    df['Hour'] = df['Meter Datetime'].dt.hour
    
    anomalous_meters = ['AES2020896472402', 'KFM2020660044515', 'KFM2020660037773']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    night_hours = list(range(21, 24)) + list(range(0, 5))
    
    for i, meter_id in enumerate(anomalous_meters):
        meter_data = df[df['HES Meter Id'] == meter_id]
        if len(meter_data) == 0:
            continue
            
        hourly_avg = meter_data.groupby('Hour')['Import active power (QI+QIV)[W]'].mean()
        colors = ['red' if hour in night_hours else 'blue' for hour in hourly_avg.index]
        
        axes[i].bar(hourly_avg.index, hourly_avg.values, color=colors, alpha=0.7)
        axes[i].set_title(f'Meter {meter_id[-8:]}')
        axes[i].set_xlabel('Hour of Day')
        axes[i].set_ylabel('Average Power (W)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(0, 24, 4))
    
    plt.suptitle('Comparison of Top 3 Anomalous Meters - Hourly Consumption', fontsize=14)
    plt.tight_layout()
    plt.savefig('top_anomalous_meters_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plot saved to: top_anomalous_meters_comparison.png")