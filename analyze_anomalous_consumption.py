import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load the combined CSV data and analyze consumption patterns"""
    
    print("Loading combined load profile data...")
    # Load data in chunks to manage memory
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv('combined_load_profile_electrical.csv', chunksize=chunk_size):
        # Convert datetime columns
        chunk['Entry Datetime'] = pd.to_datetime(chunk['Entry Datetime'])
        chunk['Meter Datetime'] = pd.to_datetime(chunk['Meter Datetime'])
        
        # Extract hour from meter datetime for analysis
        chunk['Hour'] = chunk['Meter Datetime'].dt.hour
        chunk['Date'] = chunk['Meter Datetime'].dt.date
        
        chunks.append(chunk)
        
        if len(chunks) >= 5:  # Limit to first 500k records for initial analysis
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"Data loaded: {len(df):,} records")
    print(f"Date range: {df['Meter Datetime'].min()} to {df['Meter Datetime'].max()}")
    print(f"Unique meters: {df['HES Meter Id'].nunique()}")
    
    return df

def analyze_hourly_patterns(df):
    """Analyze hourly consumption patterns for each meter"""
    
    print("\nAnalyzing hourly consumption patterns...")
    
    # Calculate hourly statistics for each meter
    hourly_stats = df.groupby(['HES Meter Id', 'Hour'])['Import active power (QI+QIV)[W]'].agg([
        'mean', 'median', 'max', 'min', 'std', 'count'
    ]).reset_index()
    
    # Calculate overall statistics for each meter
    meter_stats = df.groupby('HES Meter Id')['Import active power (QI+QIV)[W]'].agg([
        'mean', 'median', 'max', 'min', 'std'
    ]).reset_index()
    
    return hourly_stats, meter_stats

def identify_anomalous_meters(hourly_stats, meter_stats):
    """Identify meters with anomalous nighttime consumption (9 PM - 4 AM)"""
    
    print("\nIdentifying anomalous nighttime consumption patterns...")
    
    # Define nighttime hours (9 PM to 4 AM)
    night_hours = list(range(21, 24)) + list(range(0, 5))  # 21, 22, 23, 0, 1, 2, 3, 4
    
    # Calculate nighttime vs daytime consumption for each meter
    anomalous_meters = []
    
    for meter_id in hourly_stats['HES Meter Id'].unique():
        meter_hourly = hourly_stats[hourly_stats['HES Meter Id'] == meter_id]
        meter_overall = meter_stats[meter_stats['HES Meter Id'] == meter_id].iloc[0]
        
        # Get nighttime and daytime consumption
        night_data = meter_hourly[meter_hourly['Hour'].isin(night_hours)]
        day_data = meter_hourly[~meter_hourly['Hour'].isin(night_hours)]
        
        if len(night_data) == 0 or len(day_data) == 0:
            continue
        
        # Calculate metrics
        night_avg = night_data['mean'].mean()
        day_avg = day_data['mean'].mean()
        night_max = night_data['mean'].max()
        day_max = day_data['mean'].max()
        night_min = night_data['mean'].min()
        overall_max = meter_overall['max']
        overall_min = meter_overall['min']
        overall_mean = meter_overall['mean']
        
        # Anomaly detection criteria
        anomaly_score = 0
        anomaly_reasons = []
        
        # 1. High nighttime consumption compared to peak
        if night_max > 0.7 * overall_max:
            anomaly_score += 3
            anomaly_reasons.append(f"High night peak: {night_max:.0f}W vs overall max {overall_max:.0f}W")
        
        # 2. Nighttime consumption higher than daytime average
        if night_avg > day_avg:
            anomaly_score += 2
            anomaly_reasons.append(f"Night avg ({night_avg:.0f}W) > Day avg ({day_avg:.0f}W)")
        
        # 3. High base consumption during night (doesn't dip close to minimum)
        if night_min > 0.5 * overall_mean:
            anomaly_score += 2
            anomaly_reasons.append(f"High night minimum: {night_min:.0f}W vs overall mean {overall_mean:.0f}W")
        
        # 4. Check for increasing consumption during night hours
        night_consumption_trend = night_data.sort_values('Hour')['mean'].values
        if len(night_consumption_trend) > 2:
            # Check if consumption is generally increasing during night
            increasing_trend = np.polyfit(range(len(night_consumption_trend)), night_consumption_trend, 1)[0]
            if increasing_trend > 10:  # Significant positive trend
                anomaly_score += 1
                anomaly_reasons.append(f"Increasing night trend: +{increasing_trend:.1f}W/hour")
        
        if anomaly_score >= 2:  # Threshold for anomalous behavior
            anomalous_meters.append({
                'meter_id': meter_id,
                'anomaly_score': anomaly_score,
                'night_avg': night_avg,
                'day_avg': day_avg,
                'night_max': night_max,
                'night_min': night_min,
                'overall_max': overall_max,
                'overall_min': overall_min,
                'overall_mean': overall_mean,
                'reasons': anomaly_reasons
            })
    
    # Sort by anomaly score
    anomalous_meters = sorted(anomalous_meters, key=lambda x: x['anomaly_score'], reverse=True)
    
    print(f"Found {len(anomalous_meters)} meters with anomalous nighttime consumption")
    
    # Display top anomalous meters
    for i, meter in enumerate(anomalous_meters[:5]):
        print(f"\n{i+1}. Meter: {meter['meter_id']}")
        print(f"   Anomaly Score: {meter['anomaly_score']}")
        print(f"   Night avg: {meter['night_avg']:.0f}W, Day avg: {meter['day_avg']:.0f}W")
        print(f"   Reasons: {'; '.join(meter['reasons'])}")
    
    return anomalous_meters

def plot_anomalous_consumption(df, anomalous_meters, top_n=3):
    """Create plots for the most anomalous meters"""
    
    print(f"\nCreating plots for top {top_n} anomalous meters...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots for multiple meters
    fig, axes = plt.subplots(top_n, 2, figsize=(16, 6*top_n))
    if top_n == 1:
        axes = axes.reshape(1, -1)
    
    night_hours = list(range(21, 24)) + list(range(0, 5))
    
    for i in range(min(top_n, len(anomalous_meters))):
        meter_id = anomalous_meters[i]['meter_id']
        meter_data = df[df['HES Meter Id'] == meter_id].copy()
        
        if len(meter_data) == 0:
            continue
        
        # Plot 1: 24-hour average consumption pattern
        ax1 = axes[i, 0]
        hourly_avg = meter_data.groupby('Hour')['Import active power (QI+QIV)[W]'].mean()
        
        # Color nighttime hours differently
        colors = ['red' if hour in night_hours else 'blue' for hour in hourly_avg.index]
        bars = ax1.bar(hourly_avg.index, hourly_avg.values, color=colors, alpha=0.7)
        
        ax1.set_title(f'Meter {meter_id}: Average Hourly Consumption\nAnomaly Score: {anomalous_meters[i]["anomaly_score"]}')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Power (W)')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Night (9PM-4AM)'),
                          Patch(facecolor='blue', alpha=0.7, label='Day')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Time series for a sample period
        ax2 = axes[i, 1]
        
        # Get a representative week of data
        sample_data = meter_data.sample(min(1000, len(meter_data))).sort_values('Meter Datetime')
        
        ax2.plot(sample_data['Meter Datetime'], sample_data['Import active power (QI+QIV)[W]'], 
                alpha=0.7, linewidth=0.8)
        ax2.set_title(f'Meter {meter_id}: Sample Time Series')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Power (W)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Highlight nighttime consumption
        night_mask = sample_data['Hour'].isin(night_hours)
        night_data = sample_data[night_mask]
        if len(night_data) > 0:
            ax2.scatter(night_data['Meter Datetime'], night_data['Import active power (QI+QIV)[W]'], 
                       color='red', alpha=0.6, s=10, label='Night consumption')
            ax2.legend()
    
    plt.tight_layout()
    plt.savefig('anomalous_consumption_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary heatmap
    create_consumption_heatmap(df, anomalous_meters[:5])

def create_consumption_heatmap(df, anomalous_meters):
    """Create a heatmap showing consumption patterns for anomalous meters"""
    
    print("Creating consumption heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    meter_labels = []
    
    for meter_info in anomalous_meters:
        meter_id = meter_info['meter_id']
        meter_data = df[df['HES Meter Id'] == meter_id]
        
        if len(meter_data) == 0:
            continue
        
        # Calculate hourly averages
        hourly_avg = meter_data.groupby('Hour')['Import active power (QI+QIV)[W]'].mean()
        
        # Ensure we have all 24 hours
        full_hourly = pd.Series(index=range(24), dtype=float)
        full_hourly.update(hourly_avg)
        full_hourly = full_hourly.fillna(0)
        
        heatmap_data.append(full_hourly.values)
        meter_labels.append(f"{meter_id[-8:]}")  # Last 8 chars for readability
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        sns.heatmap(heatmap_array, 
                   xticklabels=range(24),
                   yticklabels=meter_labels,
                   annot=False,
                   fmt='.0f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Power Consumption (W)'},
                   ax=ax)
        
        ax.set_title('Hourly Consumption Heatmap - Anomalous Meters')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Meter ID (last 8 digits)')
        
        # Highlight nighttime hours
        night_hours = list(range(21, 24)) + list(range(0, 5))
        for hour in night_hours:
            ax.axvline(x=hour+0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig('consumption_heatmap_anomalous_meters.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function"""
    
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Analyze hourly patterns
    hourly_stats, meter_stats = analyze_hourly_patterns(df)
    
    # Identify anomalous meters
    anomalous_meters = identify_anomalous_meters(hourly_stats, meter_stats)
    
    if anomalous_meters:
        # Create plots
        plot_anomalous_consumption(df, anomalous_meters, top_n=3)
        
        # Save results
        results_df = pd.DataFrame(anomalous_meters)
        results_df.to_csv('anomalous_meters_analysis.csv', index=False)
        print(f"\nResults saved to: anomalous_meters_analysis.csv")
        print(f"Plots saved to: anomalous_consumption_analysis.png and consumption_heatmap_anomalous_meters.png")
    else:
        print("No significantly anomalous meters found.")

if __name__ == "__main__":
    main()