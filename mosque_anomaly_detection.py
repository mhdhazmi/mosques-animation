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

class MosqueAnomalyDetector:
    """
    Comprehensive anomaly detection for mosque smart meter data
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
        
    def load_data(self, file_path):
        """Load parquet file and prepare data"""
        print("Loading mosque consumption data...")
        
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        return df
    
    def prepare_features(self, df, datetime_col, consumption_col, meter_col=None):
        """Extract time-based features for anomaly detection"""
        print("Preparing features...")
        
        # Ensure datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Extract time features
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)  # Friday is special
        
        # Prayer time features
        df['is_prayer_time'] = df['hour'].apply(self._is_prayer_time)
        df['prayer_period'] = df['hour'].apply(self._get_prayer_period)
        df['is_low_hour'] = df['hour'].isin(self.low_consumption_hours).astype(int)
        
        # Consumption features
        if meter_col:
            # Calculate rolling statistics per meter
            df = df.sort_values([meter_col, datetime_col])
            df['consumption_rolling_mean'] = df.groupby(meter_col)[consumption_col].transform(
                lambda x: x.rolling(window=48, min_periods=1).mean()
            )
            df['consumption_rolling_std'] = df.groupby(meter_col)[consumption_col].transform(
                lambda x: x.rolling(window=48, min_periods=1).std()
            )
        else:
            # Single meter
            df = df.sort_values(datetime_col)
            df['consumption_rolling_mean'] = df[consumption_col].rolling(window=48, min_periods=1).mean()
            df['consumption_rolling_std'] = df[consumption_col].rolling(window=48, min_periods=1).std()
        
        # Normalize consumption relative to rolling average
        df['consumption_normalized'] = (df[consumption_col] - df['consumption_rolling_mean']) / (df['consumption_rolling_std'] + 1e-6)
        
        return df
    
    def _is_prayer_time(self, hour):
        """Check if hour falls within prayer time"""
        for prayer, (start, end) in self.prayer_times.items():
            if start <= hour < end:
                return 1
        return 0
    
    def _get_prayer_period(self, hour):
        """Get prayer period name"""
        for prayer, (start, end) in self.prayer_times.items():
            if start <= hour < end:
                return prayer
        return 'Non-prayer'
    
    def detect_statistical_anomalies(self, df, consumption_col, threshold=3):
        """Detect anomalies using statistical methods"""
        print("Detecting statistical anomalies...")
        
        anomalies = {}
        
        # 1. Z-Score method
        z_scores = np.abs(stats.zscore(df[consumption_col].fillna(df[consumption_col].mean())))
        anomalies['z_score'] = z_scores > threshold
        
        # 2. Modified Z-Score (more robust)
        median = df[consumption_col].median()
        mad = np.median(np.abs(df[consumption_col] - median))
        modified_z_scores = 0.6745 * (df[consumption_col] - median) / (mad + 1e-6)
        anomalies['modified_z_score'] = np.abs(modified_z_scores) > threshold
        
        # 3. IQR method
        Q1 = df[consumption_col].quantile(0.25)
        Q3 = df[consumption_col].quantile(0.75)
        IQR = Q3 - Q1
        anomalies['iqr'] = (df[consumption_col] < (Q1 - 1.5 * IQR)) | (df[consumption_col] > (Q3 + 1.5 * IQR))
        
        return anomalies
    
    def detect_contextual_anomalies(self, df, consumption_col):
        """Detect context-specific anomalies for mosques"""
        print("Detecting contextual anomalies...")
        
        anomalies = {}
        
        # 1. High consumption during non-prayer hours
        non_prayer_mask = df['is_prayer_time'] == 0
        non_prayer_threshold = df[non_prayer_mask][consumption_col].quantile(0.95)
        anomalies['high_non_prayer'] = non_prayer_mask & (df[consumption_col] > non_prayer_threshold)
        
        # 2. Low consumption during prayer times
        prayer_mask = df['is_prayer_time'] == 1
        prayer_threshold = df[prayer_mask][consumption_col].quantile(0.25)
        anomalies['low_prayer'] = prayer_mask & (df[consumption_col] < prayer_threshold)
        
        # 3. No Friday spike
        friday_mask = df['is_friday'] == 1
        friday_dhuhr_mask = friday_mask & (df['prayer_period'] == 'Dhuhr')
        if friday_dhuhr_mask.any():
            friday_avg = df[friday_dhuhr_mask][consumption_col].mean()
            normal_dhuhr_avg = df[(df['is_friday'] == 0) & (df['prayer_period'] == 'Dhuhr')][consumption_col].mean()
            anomalies['no_friday_spike'] = friday_dhuhr_mask & (df[consumption_col] < normal_dhuhr_avg * 1.5)
        
        # 4. Night consumption anomalies (2 AM - 4 AM should be minimal)
        night_mask = df['hour'].isin([2, 3])
        night_threshold = df[consumption_col].quantile(0.10)
        anomalies['high_night'] = night_mask & (df[consumption_col] > night_threshold * 3)
        
        return anomalies
    
    def detect_ml_anomalies(self, df, feature_cols, contamination=0.1):
        """Use Isolation Forest for anomaly detection"""
        print("Detecting ML-based anomalies...")
        
        # Prepare features
        features = df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(features_scaled)
        
        # -1 indicates anomaly, 1 indicates normal
        anomalies = predictions == -1
        
        # Get anomaly scores
        scores = iso_forest.score_samples(features_scaled)
        
        return anomalies, scores
    
    def analyze_anomaly_patterns(self, df, anomalies_dict, consumption_col):
        """Analyze and summarize anomaly patterns"""
        print("\nAnomaly Detection Summary:")
        print("=" * 50)
        
        total_records = len(df)
        summary = {}
        
        for method, anomaly_mask in anomalies_dict.items():
            count = anomaly_mask.sum()
            percentage = (count / total_records) * 100
            
            summary[method] = {
                'count': count,
                'percentage': percentage,
                'mean_consumption': df[anomaly_mask][consumption_col].mean() if count > 0 else 0,
                'std_consumption': df[anomaly_mask][consumption_col].std() if count > 0 else 0
            }
            
            print(f"\n{method}:")
            print(f"  Anomalies found: {count} ({percentage:.2f}%)")
            if count > 0:
                print(f"  Mean consumption: {summary[method]['mean_consumption']:.2f}")
                print(f"  Std consumption: {summary[method]['std_consumption']:.2f}")
        
        return summary
    
    def create_anomaly_report(self, df, anomalies_dict, consumption_col, datetime_col, 
                            meter_col=None, save_path='mosque_anomaly_report.html'):
        """Create comprehensive anomaly report with visualizations"""
        print("\nCreating anomaly report...")
        
        # Combine all anomalies
        df['any_anomaly'] = pd.DataFrame(list(anomalies_dict.values())).any(axis=0).values
        df['anomaly_count'] = pd.DataFrame(list(anomalies_dict.values())).sum(axis=0).values
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Hourly pattern with anomalies
        ax1 = axes[0, 0]
        hourly_normal = df[~df['any_anomaly']].groupby('hour')[consumption_col].mean()
        hourly_anomaly = df[df['any_anomaly']].groupby('hour')[consumption_col].mean()
        
        ax1.bar(hourly_normal.index, hourly_normal.values, alpha=0.7, label='Normal', color='blue')
        ax1.bar(hourly_anomaly.index, hourly_anomaly.values, alpha=0.7, label='Anomaly', color='red')
        
        # Highlight prayer times
        for prayer, (start, end) in self.prayer_times.items():
            ax1.axvspan(start, end, alpha=0.2, color='green', label=prayer if start == 4 else '')
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Consumption')
        ax1.set_title('Hourly Consumption Pattern: Normal vs Anomaly')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Anomaly distribution by type
        ax2 = axes[0, 1]
        anomaly_counts = {method: mask.sum() for method, mask in anomalies_dict.items()}
        
        ax2.bar(anomaly_counts.keys(), anomaly_counts.values(), color='darkred')
        ax2.set_xlabel('Anomaly Detection Method')
        ax2.set_ylabel('Count')
        ax2.set_title('Anomalies by Detection Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Time series with anomalies
        ax3 = axes[1, 0]
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size).sort_values(datetime_col)
        
        normal_mask = ~sample_df['any_anomaly']
        anomaly_mask = sample_df['any_anomaly']
        
        ax3.scatter(sample_df[normal_mask][datetime_col], 
                   sample_df[normal_mask][consumption_col], 
                   alpha=0.5, s=10, label='Normal', color='blue')
        ax3.scatter(sample_df[anomaly_mask][datetime_col], 
                   sample_df[anomaly_mask][consumption_col], 
                   alpha=0.8, s=30, label='Anomaly', color='red', marker='x')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Consumption')
        ax3.set_title('Consumption Time Series with Anomalies (Sample)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Prayer time vs non-prayer time anomalies
        ax4 = axes[1, 1]
        prayer_anomaly_stats = df.groupby(['is_prayer_time', 'any_anomaly']).size().unstack(fill_value=0)
        
        prayer_anomaly_stats.plot(kind='bar', ax=ax4, color=['blue', 'red'])
        ax4.set_xlabel('Is Prayer Time')
        ax4.set_ylabel('Count')
        ax4.set_title('Anomaly Distribution: Prayer vs Non-Prayer Hours')
        ax4.set_xticklabels(['Non-Prayer Hours', 'Prayer Hours'], rotation=0)
        ax4.legend(['Normal', 'Anomaly'])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mosque_anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save anomalous records
        anomalous_records = df[df['any_anomaly']]
        anomalous_records.to_csv('mosque_anomalous_records.csv', index=False)
        
        print(f"\nReport saved:")
        print(f"  - Plot: mosque_anomaly_analysis.png")
        print(f"  - Anomalous records: mosque_anomalous_records.csv")
        
        return df

# Example usage
def run_mosque_anomaly_detection(file_path, datetime_col, consumption_col, meter_col=None):
    """Run complete anomaly detection pipeline"""
    
    detector = MosqueAnomalyDetector()
    
    # Load data
    df = detector.load_data(file_path)
    
    # Prepare features
    df = detector.prepare_features(df, datetime_col, consumption_col, meter_col)
    
    # Statistical anomalies
    stat_anomalies = detector.detect_statistical_anomalies(df, consumption_col)
    
    # Contextual anomalies
    context_anomalies = detector.detect_contextual_anomalies(df, consumption_col)
    
    # ML-based anomalies
    feature_cols = ['hour', 'day_of_week', 'is_friday', 'is_prayer_time', 
                    'consumption_normalized', consumption_col]
    ml_anomalies, anomaly_scores = detector.detect_ml_anomalies(df, feature_cols)
    
    # Combine all anomalies
    all_anomalies = {**stat_anomalies, **context_anomalies, 'ml_isolation_forest': ml_anomalies}
    
    # Analyze patterns
    summary = detector.analyze_anomaly_patterns(df, all_anomalies, consumption_col)
    
    # Create report
    df_with_anomalies = detector.create_anomaly_report(df, all_anomalies, consumption_col, 
                                                      datetime_col, meter_col)
    
    return df_with_anomalies, summary

if __name__ == "__main__":
    # Example usage with the parquet file
    file_path = 'center_east_Mousq_2_Filttered_with_Adhan_All_2.parquet'
    
    # You'll need to adjust these column names based on your actual data
    # Common column names in smart meter data:
    datetime_col = 'timestamp'  # or 'Meter Datetime', 'datetime', etc.
    consumption_col = 'consumption'  # or 'Import active power (QI+QIV)[W]', 'kWh', etc.
    meter_col = 'meter_id'  # or 'HES Meter Id', 'device_id', etc. (set to None if single meter)
    
    try:
        df_results, summary = run_mosque_anomaly_detection(
            file_path, datetime_col, consumption_col, meter_col
        )
        print("\nAnomaly detection completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check your column names and try again.")