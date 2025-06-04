import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MosqueAnomalyDetectorChunked:
    """
    Memory-efficient anomaly detection for large mosque smart meter datasets
    """
    
    def __init__(self):
        # Prayer times (approximate, would need adjustment based on location/season)
        self.prayer_times = {
            'Fajr': (4, 6),      # 4 AM - 6 AM
            'Dhuhr': (12, 14),   # 12 PM - 2 PM
            'Asr': (15, 17),     # 3 PM - 5 PM
            'Maghrib': (18, 20), # 6 PM - 8 PM
            'Isha': (20, 22)     # 8 PM - 10 PM
        }
        
        # Expected low consumption hours (between prayers)
        self.low_consumption_hours = [2, 3, 8, 9, 10, 11, 23]
        
    def analyze_sample_meters(self, file_path, n_meters=5, days_per_meter=30):
        """Analyze a sample of meters to avoid memory issues"""
        print(f"Analyzing sample of {n_meters} meters for {days_per_meter} days each...")
        
        # First, get unique meter IDs
        print("Getting unique meter IDs...")
        meter_ids = pd.read_parquet(file_path, columns=['METER_ID'])['METER_ID'].unique()
        print(f"Total unique meters: {len(meter_ids)}")
        
        # Sample meters
        if len(meter_ids) > n_meters:
            sample_meter_ids = np.random.choice(meter_ids, n_meters, replace=False)
        else:
            sample_meter_ids = meter_ids
        
        all_results = []
        anomaly_summaries = []
        
        for i, meter_id in enumerate(sample_meter_ids):
            print(f"\nAnalyzing meter {i+1}/{len(sample_meter_ids)}: {meter_id}")
            
            # Load data for this specific meter
            df_meter = pd.read_parquet(
                file_path,
                filters=[('METER_ID', '==', meter_id)]
            )
            
            # Limit to specified number of days
            df_meter = df_meter.sort_values('READING_DATETIME')
            if len(df_meter) > days_per_meter * 48:  # Assuming 30-minute intervals
                df_meter = df_meter.iloc[:days_per_meter * 48]
            
            # Run anomaly detection for this meter
            meter_results = self.analyze_single_meter(df_meter, meter_id)
            all_results.append(meter_results)
            
            # Collect summary statistics
            if meter_results['anomalies_found']:
                anomaly_summaries.append({
                    'meter_id': meter_id,
                    'total_records': len(df_meter),
                    'anomaly_count': meter_results['anomaly_count'],
                    'anomaly_percentage': meter_results['anomaly_percentage'],
                    'high_night_count': meter_results['high_night_count'],
                    'missing_prayer_count': meter_results['missing_prayer_count']
                })
        
        # Create summary report
        self.create_summary_report(anomaly_summaries, sample_meter_ids)
        
        return all_results
    
    def analyze_single_meter(self, df, meter_id):
        """Analyze a single meter's data"""
        
        # Prepare features
        df = df.copy()
        df['hour'] = df['READING_DATETIME'].dt.hour
        df['day_of_week'] = df['READING_DATETIME'].dt.dayofweek
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Prayer time features
        df['is_prayer_time'] = df['hour'].apply(self._is_prayer_time)
        df['is_low_hour'] = df['hour'].isin(self.low_consumption_hours).astype(int)
        
        # Calculate rolling statistics
        df['consumption_rolling_mean'] = df['ACTIVE_IMP_POWER'].rolling(window=48, min_periods=1).mean()
        df['consumption_rolling_std'] = df['ACTIVE_IMP_POWER'].rolling(window=48, min_periods=1).std()
        
        # Detect anomalies
        anomalies = self.detect_meter_anomalies(df)
        
        # Calculate summary statistics
        total_anomalies = anomalies['total_anomalies'].sum()
        anomaly_percentage = (total_anomalies / len(df)) * 100
        
        # Count specific types of anomalies
        high_night_count = anomalies['high_night'].sum()
        missing_prayer_count = anomalies['low_prayer'].sum()
        
        results = {
            'meter_id': meter_id,
            'data': df,
            'anomalies': anomalies,
            'anomalies_found': total_anomalies > 0,
            'anomaly_count': total_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'high_night_count': high_night_count,
            'missing_prayer_count': missing_prayer_count
        }
        
        return results
    
    def _is_prayer_time(self, hour):
        """Check if hour falls within prayer time"""
        for prayer, (start, end) in self.prayer_times.items():
            if start <= hour < end:
                return 1
        return 0
    
    def detect_meter_anomalies(self, df):
        """Detect various types of anomalies"""
        anomalies = pd.DataFrame(index=df.index)
        
        # 1. Statistical anomalies (Z-score)
        z_scores = np.abs(stats.zscore(df['ACTIVE_IMP_POWER'].fillna(df['ACTIVE_IMP_POWER'].mean())))
        anomalies['z_score'] = z_scores > 3
        
        # 2. IQR anomalies
        Q1 = df['ACTIVE_IMP_POWER'].quantile(0.25)
        Q3 = df['ACTIVE_IMP_POWER'].quantile(0.75)
        IQR = Q3 - Q1
        anomalies['iqr'] = (df['ACTIVE_IMP_POWER'] < (Q1 - 1.5 * IQR)) | (df['ACTIVE_IMP_POWER'] > (Q3 + 1.5 * IQR))
        
        # 3. High night consumption (2-4 AM)
        night_mask = df['hour'].isin([2, 3])
        night_threshold = df['ACTIVE_IMP_POWER'].quantile(0.90)
        anomalies['high_night'] = night_mask & (df['ACTIVE_IMP_POWER'] > night_threshold)
        
        # 4. Low consumption during prayer times
        prayer_mask = df['is_prayer_time'] == 1
        if prayer_mask.any():
            prayer_threshold = df[prayer_mask]['ACTIVE_IMP_POWER'].quantile(0.25)
            anomalies['low_prayer'] = prayer_mask & (df['ACTIVE_IMP_POWER'] < prayer_threshold)
        else:
            anomalies['low_prayer'] = False
        
        # 5. No Friday spike during Dhuhr
        friday_dhuhr = (df['is_friday'] == 1) & (df['hour'].between(12, 14))
        if friday_dhuhr.any():
            friday_avg = df[friday_dhuhr]['ACTIVE_IMP_POWER'].mean()
            normal_dhuhr_avg = df[(df['is_friday'] == 0) & (df['hour'].between(12, 14))]['ACTIVE_IMP_POWER'].mean()
            anomalies['no_friday_spike'] = friday_dhuhr & (df['ACTIVE_IMP_POWER'] < normal_dhuhr_avg * 1.2)
        else:
            anomalies['no_friday_spike'] = False
        
        # Total anomalies
        anomalies['total_anomalies'] = anomalies.sum(axis=1)
        
        return anomalies
    
    def create_summary_report(self, anomaly_summaries, sample_meter_ids):
        """Create visualization summary for all analyzed meters"""
        
        if not anomaly_summaries:
            print("No anomalies found in the sampled meters")
            return
        
        # Convert to DataFrame for easier analysis
        summary_df = pd.DataFrame(anomaly_summaries)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Anomaly percentage by meter
        ax1 = axes[0, 0]
        meters = summary_df['meter_id'].apply(lambda x: x[-8:])  # Last 8 chars
        ax1.bar(meters, summary_df['anomaly_percentage'], color='darkred')
        ax1.set_xlabel('Meter ID (last 8 digits)')
        ax1.set_ylabel('Anomaly Percentage (%)')
        ax1.set_title('Anomaly Rate by Meter')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Types of anomalies distribution
        ax2 = axes[0, 1]
        anomaly_types = ['High Night', 'Missing Prayer']
        type_counts = [
            summary_df['high_night_count'].sum(),
            summary_df['missing_prayer_count'].sum()
        ]
        ax2.bar(anomaly_types, type_counts, color=['darkblue', 'darkgreen'])
        ax2.set_xlabel('Anomaly Type')
        ax2.set_ylabel('Total Count')
        ax2.set_title('Distribution of Anomaly Types')
        ax2.grid(True, alpha=0.3)
        
        # 3. Anomaly count distribution
        ax3 = axes[1, 0]
        ax3.hist(summary_df['anomaly_count'], bins=20, color='purple', alpha=0.7)
        ax3.set_xlabel('Number of Anomalies')
        ax3.set_ylabel('Number of Meters')
        ax3.set_title('Distribution of Anomaly Counts')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
        Mosque Anomaly Detection Summary
        ================================
        
        Total Meters Analyzed: {len(sample_meter_ids)}
        Meters with Anomalies: {len(summary_df)}
        
        Average Anomaly Rate: {summary_df['anomaly_percentage'].mean():.2f}%
        Max Anomaly Rate: {summary_df['anomaly_percentage'].max():.2f}%
        
        Total Anomalies Found: {summary_df['anomaly_count'].sum()}
        - High Night Consumption: {summary_df['high_night_count'].sum()}
        - Missing Prayer Peaks: {summary_df['missing_prayer_count'].sum()}
        
        Most Anomalous Meter: {summary_df.loc[summary_df['anomaly_percentage'].idxmax(), 'meter_id']}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('mosque_anomaly_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        summary_df.to_csv('mosque_anomaly_details.csv', index=False)
        
        print(f"\nReports saved:")
        print(f"  - Summary plot: mosque_anomaly_summary.png")
        print(f"  - Detailed results: mosque_anomaly_details.csv")
        
        # Print top anomalous meters
        print("\nTop 5 Most Anomalous Meters:")
        top_meters = summary_df.nlargest(5, 'anomaly_percentage')
        for idx, row in top_meters.iterrows():
            print(f"  {row['meter_id']}: {row['anomaly_percentage']:.2f}% anomalies")
    
    def analyze_prayer_time_patterns(self, file_path, meter_id):
        """Analyze consumption patterns around prayer times for a specific meter"""
        print(f"\nAnalyzing prayer time patterns for meter {meter_id}...")
        
        # Load data for specific meter
        df = pd.read_parquet(
            file_path,
            filters=[('METER_ID', '==', meter_id)]
        )
        
        df['hour'] = df['READING_DATETIME'].dt.hour
        df['day_of_week'] = df['READING_DATETIME'].dt.dayofweek
        
        # Calculate average consumption by hour
        hourly_avg = df.groupby('hour')['ACTIVE_IMP_POWER'].mean()
        
        # Create prayer time visualization
        plt.figure(figsize=(12, 6))
        
        # Plot hourly consumption
        plt.bar(hourly_avg.index, hourly_avg.values, alpha=0.7, color='blue')
        
        # Highlight prayer times
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (prayer, (start, end)) in enumerate(self.prayer_times.items()):
            plt.axvspan(start, end, alpha=0.2, color=colors[i], label=prayer)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Power Consumption (W)')
        plt.title(f'Meter {meter_id}: Hourly Consumption with Prayer Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        
        plt.tight_layout()
        plt.savefig(f'prayer_pattern_{meter_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prayer pattern plot saved: prayer_pattern_{meter_id}.png")
        
        # Calculate prayer vs non-prayer consumption
        prayer_hours = []
        for start, end in self.prayer_times.values():
            prayer_hours.extend(range(start, end))
        
        prayer_consumption = df[df['hour'].isin(prayer_hours)]['ACTIVE_IMP_POWER'].mean()
        non_prayer_consumption = df[~df['hour'].isin(prayer_hours)]['ACTIVE_IMP_POWER'].mean()
        
        print(f"\nConsumption Analysis:")
        print(f"  Average during prayer times: {prayer_consumption:.2f} W")
        print(f"  Average during non-prayer times: {non_prayer_consumption:.2f} W")
        print(f"  Prayer/Non-prayer ratio: {prayer_consumption/non_prayer_consumption:.2f}")

def run_chunked_analysis(file_path='center_east_Mousq_2_Filttered_with_Adhan_All_2.parquet'):
    """Run memory-efficient anomaly detection on large dataset"""
    
    detector = MosqueAnomalyDetectorChunked()
    
    # Analyze a sample of meters
    results = detector.analyze_sample_meters(
        file_path,
        n_meters=10,  # Analyze 10 random meters
        days_per_meter=30  # 30 days per meter
    )
    
    # If anomalies found, analyze prayer patterns for most anomalous meter
    if results:
        # Find meter with highest anomaly rate
        max_anomaly_meter = None
        max_anomaly_rate = 0
        
        for result in results:
            if result['anomaly_percentage'] > max_anomaly_rate:
                max_anomaly_rate = result['anomaly_percentage']
                max_anomaly_meter = result['meter_id']
        
        if max_anomaly_meter:
            detector.analyze_prayer_time_patterns(file_path, max_anomaly_meter)
    
    return results

if __name__ == "__main__":
    run_chunked_analysis()