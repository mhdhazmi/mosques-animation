import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MosqueThresholdAnalyzer:
    """
    Analyze and recommend thresholds for night/morning violation detection
    """
    
    def __init__(self):
        # Define time periods
        self.time_periods = {
            'deep_night': (23, 3),      # 11 PM - 3 AM (minimal activity expected)
            'pre_fajr': (3, 4),         # 3 AM - 4 AM (some preparation activity)
            'fajr_time': (4, 6),        # 4 AM - 6 AM (prayer time)
            'morning_low': (6, 7),      # 6 AM - 7 AM (post-Fajr low)
        }
        
        # Essential equipment that runs 24/7
        self.baseline_equipment = {
            'emergency_lighting': 50,    # Watts
            'security_system': 30,       # Watts
            'refrigeration': 150,        # Watts (if morgue facility)
            'hvac_standby': 100,        # Watts (ventilation)
        }
    
    def analyze_night_patterns(self, file_path, sample_size=50):
        """Analyze night consumption patterns across multiple mosques"""
        print("Analyzing night consumption patterns...")
        
        # Get unique meter IDs
        meter_ids = pd.read_parquet(file_path, columns=['METER_ID'])['METER_ID'].unique()
        
        # Sample meters
        if len(meter_ids) > sample_size:
            sample_meters = np.random.choice(meter_ids, sample_size, replace=False)
        else:
            sample_meters = meter_ids
        
        night_consumption_data = []
        
        for meter_id in sample_meters:
            # Load data for this meter
            df = pd.read_parquet(
                file_path,
                filters=[('METER_ID', '==', meter_id)]
            )
            
            df['hour'] = df['READING_DATETIME'].dt.hour
            df['month'] = df['READING_DATETIME'].dt.month
            
            # Analyze each time period
            for period_name, (start_hour, end_hour) in self.time_periods.items():
                if start_hour > end_hour:  # Handle day boundary
                    mask = (df['hour'] >= start_hour) | (df['hour'] < end_hour)
                else:
                    mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
                
                period_data = df[mask]['ACTIVE_IMP_POWER']
                
                if len(period_data) > 0:
                    night_consumption_data.append({
                        'meter_id': meter_id,
                        'period': period_name,
                        'mean': period_data.mean(),
                        'median': period_data.median(),
                        'std': period_data.std(),
                        'min': period_data.min(),
                        'max': period_data.max(),
                        'p10': period_data.quantile(0.10),
                        'p25': period_data.quantile(0.25),
                        'p75': period_data.quantile(0.75),
                        'p90': period_data.quantile(0.90),
                        'p95': period_data.quantile(0.95),
                        'p99': period_data.quantile(0.99),
                        'count': len(period_data)
                    })
        
        # Convert to DataFrame
        night_df = pd.DataFrame(night_consumption_data)
        
        return night_df
    
    def calculate_baseline_consumption(self, mosque_size='medium'):
        """Calculate expected baseline consumption based on mosque size"""
        
        size_multipliers = {
            'small': 0.7,
            'medium': 1.0,
            'large': 1.5,
            'very_large': 2.0
        }
        
        multiplier = size_multipliers.get(mosque_size, 1.0)
        
        baseline = sum(self.baseline_equipment.values()) * multiplier
        
        return baseline
    
    def recommend_thresholds(self, night_df):
        """Recommend both static and dynamic thresholds"""
        
        recommendations = {}
        
        for period in self.time_periods.keys():
            period_data = night_df[night_df['period'] == period]
            
            if len(period_data) == 0:
                continue
            
            # Calculate statistics across all mosques
            overall_stats = {
                'mean_of_means': period_data['mean'].mean(),
                'median_of_medians': period_data['median'].median(),
                'typical_max': period_data['p95'].median(),  # Typical 95th percentile
                'absolute_max': period_data['max'].quantile(0.95),  # 95th percentile of maxes
            }
            
            # Static threshold recommendations
            if period == 'deep_night':
                # Most restrictive - only essential equipment
                static_threshold = self.calculate_baseline_consumption('large')
            elif period == 'pre_fajr':
                # Allow some preparation activity
                static_threshold = self.calculate_baseline_consumption('large') * 2
            elif period == 'fajr_time':
                # Prayer time - much higher allowed
                static_threshold = overall_stats['typical_max']
            else:
                # Morning low period
                static_threshold = self.calculate_baseline_consumption('large') * 1.5
            
            # Dynamic threshold recommendations
            dynamic_rules = {
                'baseline_method': 'rolling_median',
                'window_days': 14,  # 2 weeks of history
                'multiplier': self._get_dynamic_multiplier(period),
                'min_threshold': self.calculate_baseline_consumption('small'),
                'max_threshold': static_threshold * 1.5,
                'seasonal_adjustment': True,
                'outlier_method': 'modified_zscore',
                'outlier_threshold': 3.5
            }
            
            recommendations[period] = {
                'static_threshold': static_threshold,
                'dynamic_rules': dynamic_rules,
                'statistics': overall_stats,
                'violation_criteria': self._get_violation_criteria(period)
            }
        
        return recommendations
    
    def _get_dynamic_multiplier(self, period):
        """Get dynamic threshold multiplier based on period"""
        multipliers = {
            'deep_night': 2.0,      # 2x the rolling baseline
            'pre_fajr': 2.5,        # 2.5x the rolling baseline
            'fajr_time': 4.0,       # 4x the rolling baseline
            'morning_low': 2.5      # 2.5x the rolling baseline
        }
        return multipliers.get(period, 2.5)
    
    def _get_violation_criteria(self, period):
        """Define when a threshold violation should trigger an alert"""
        criteria = {
            'deep_night': {
                'consecutive_readings': 2,  # 2 consecutive 30-min readings
                'severity_levels': {
                    'warning': 1.5,   # 1.5x threshold
                    'alert': 2.0,     # 2x threshold
                    'critical': 3.0   # 3x threshold
                }
            },
            'pre_fajr': {
                'consecutive_readings': 3,
                'severity_levels': {
                    'warning': 1.5,
                    'alert': 2.5,
                    'critical': 4.0
                }
            },
            'fajr_time': {
                'consecutive_readings': 4,  # More tolerance during prayer
                'severity_levels': {
                    'warning': 2.0,
                    'alert': 3.0,
                    'critical': 5.0
                }
            },
            'morning_low': {
                'consecutive_readings': 3,
                'severity_levels': {
                    'warning': 1.5,
                    'alert': 2.0,
                    'critical': 3.0
                }
            }
        }
        return criteria.get(period)
    
    def visualize_threshold_recommendations(self, night_df, recommendations):
        """Create visualizations for threshold recommendations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (period, (start_hour, end_hour)) in enumerate(self.time_periods.items()):
            ax = axes[idx]
            
            period_data = night_df[night_df['period'] == period]
            rec = recommendations[period]
            
            # Plot distribution of mean consumptions
            if len(period_data) > 0:
                # Histogram of mean consumptions
                ax.hist(period_data['mean'], bins=30, alpha=0.6, color='blue', label='Mosque Averages')
                
                # Add threshold lines
                ax.axvline(rec['static_threshold'], color='red', linestyle='--', 
                          linewidth=2, label=f'Static Threshold: {rec["static_threshold"]:.0f}W')
                
                # Add baseline equipment line
                baseline = self.calculate_baseline_consumption('medium')
                ax.axvline(baseline, color='green', linestyle=':', 
                          linewidth=2, label=f'Essential Equipment: {baseline:.0f}W')
                
                # Add percentile lines
                ax.axvline(period_data['mean'].quantile(0.95), color='orange', linestyle='-', 
                          linewidth=1, label='95th Percentile')
                
                ax.set_xlabel('Average Power Consumption (W)')
                ax.set_ylabel('Number of Mosques')
                ax.set_title(f'{period.replace("_", " ").title()} ({start_hour}:00 - {end_hour}:00)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a detailed comparison plot
        self._create_threshold_comparison_plot(night_df, recommendations)
    
    def _create_threshold_comparison_plot(self, night_df, recommendations):
        """Create detailed threshold comparison visualization"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for grouped bar chart
        periods = list(self.time_periods.keys())
        static_thresholds = [recommendations[p]['static_threshold'] for p in periods]
        typical_consumption = [recommendations[p]['statistics']['median_of_medians'] for p in periods]
        max_normal = [recommendations[p]['statistics']['typical_max'] for p in periods]
        
        x = np.arange(len(periods))
        width = 0.25
        
        # Create bars
        ax.bar(x - width, typical_consumption, width, label='Typical Consumption', color='blue', alpha=0.7)
        ax.bar(x, max_normal, width, label='95th Percentile', color='orange', alpha=0.7)
        ax.bar(x + width, static_thresholds, width, label='Recommended Threshold', color='red', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Power Consumption (W)')
        ax.set_title('Threshold Recommendations by Time Period')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in periods])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(static_thresholds):
            ax.text(i + width, v + 20, f'{v:.0f}W', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_threshold_report(self, recommendations):
        """Generate a detailed threshold recommendation report"""
        
        report = """
# MOSQUE CONSUMPTION THRESHOLD RECOMMENDATIONS
==============================================

## RECOMMENDED APPROACH: HYBRID SYSTEM

### Why Hybrid (Dynamic + Static)?

1. **Dynamic Component** adapts to:
   - Seasonal variations (summer AC, winter heating)
   - Mosque size differences
   - Local consumption patterns
   - Gradual equipment changes

2. **Static Component** provides:
   - Hard safety limits
   - Regulatory compliance
   - Protection against gradual drift
   - Simple violation rules

## THRESHOLD RECOMMENDATIONS BY TIME PERIOD

"""
        
        for period, rec in recommendations.items():
            hours = self.time_periods[period]
            report += f"""
### {period.replace('_', ' ').upper()} ({hours[0]:02d}:00 - {hours[1]:02d}:00)

**Static Threshold:** {rec['static_threshold']:.0f} Watts

**Dynamic Threshold Formula:**
- Baseline = 14-day rolling median of same time period
- Threshold = Baseline × {rec['dynamic_rules']['multiplier']}
- Minimum allowed: {rec['dynamic_rules']['min_threshold']:.0f}W
- Maximum allowed: {rec['dynamic_rules']['max_threshold']:.0f}W

**Violation Detection:**
- Trigger after {rec['violation_criteria']['consecutive_readings']} consecutive readings above threshold
- Warning at {rec['violation_criteria']['severity_levels']['warning']}× threshold
- Alert at {rec['violation_criteria']['severity_levels']['alert']}× threshold  
- Critical at {rec['violation_criteria']['severity_levels']['critical']}× threshold

**Typical Consumption:** {rec['statistics']['median_of_medians']:.0f}W
**95th Percentile:** {rec['statistics']['typical_max']:.0f}W

"""
        
        report += """
## IMPLEMENTATION GUIDELINES

### 1. Initial Setup
- Use static thresholds for first 14 days
- Collect baseline data for dynamic calculations
- Classify mosques by size (small/medium/large)

### 2. Dynamic Baseline Calculation
```python
baseline = rolling_median(14_days, same_hour_readings)
if month in [6,7,8]:  # Summer
    baseline *= 1.2
elif month in [12,1,2]:  # Winter  
    baseline *= 1.1
```

### 3. Alert Priority Matrix
- Deep Night + Critical = Immediate investigation
- Pre-Fajr + Alert = Next day follow-up
- Fajr Time + Warning = Weekly review

### 4. Special Considerations
- Ramadan: Increase night thresholds by 50%
- Special events: Temporary threshold override
- Maintenance: Automatic threshold suspension

### 5. Continuous Improvement
- Monthly threshold review
- Seasonal adjustment factors
- False positive rate monitoring
- Feedback loop from field investigations
"""
        
        with open('threshold_recommendations_report.txt', 'w') as f:
            f.write(report)
        
        print("Detailed report saved to: threshold_recommendations_report.txt")
        
        return report

def analyze_and_recommend_thresholds(file_path='center_east_Mousq_2_Filttered_with_Adhan_All_2.parquet'):
    """Main function to analyze and recommend thresholds"""
    
    analyzer = MosqueThresholdAnalyzer()
    
    # Analyze night patterns
    night_df = analyzer.analyze_night_patterns(file_path, sample_size=100)
    
    # Get threshold recommendations
    recommendations = analyzer.recommend_thresholds(night_df)
    
    # Visualize recommendations
    analyzer.visualize_threshold_recommendations(night_df, recommendations)
    
    # Generate report
    report = analyzer.generate_threshold_report(recommendations)
    
    # Save processed data
    night_df.to_csv('night_consumption_analysis.csv', index=False)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- threshold_recommendations.png")
    print("- threshold_comparison.png") 
    print("- threshold_recommendations_report.txt")
    print("- night_consumption_analysis.csv")
    
    return recommendations, night_df

if __name__ == "__main__":
    recommendations, night_df = analyze_and_recommend_thresholds()