# Mosque Smart Meter Anomaly Detection Guide

## Best Approaches for Anomaly Detection

### 1. **Domain-Specific Pattern Recognition**
Mosques have unique consumption patterns tied to religious activities:

- **Prayer Time Peaks**: Consumption should spike during the 5 daily prayers
  - Fajr (dawn): 4-6 AM
  - Dhuhr (noon): 12-2 PM  
  - Asr (afternoon): 3-5 PM
  - Maghrib (sunset): 6-8 PM
  - Isha (night): 8-10 PM

- **Friday Patterns**: Expect 50-200% higher consumption during Friday prayers (Jummah)
- **Ramadan Changes**: Different patterns during fasting month (Tarawih prayers, Iftar gatherings)
- **Seasonal Variations**: AC usage in summer, heating in winter

### 2. **Statistical Methods** (Quick & Interpretable)

```python
# Z-Score Method - Detects values far from mean
z_score_threshold = 3
anomalies = np.abs(stats.zscore(consumption)) > z_score_threshold

# Modified Z-Score - More robust to outliers
median = np.median(consumption)
mad = np.median(np.abs(consumption - median))
modified_z = 0.6745 * (consumption - median) / mad
anomalies = np.abs(modified_z) > threshold

# IQR Method - Good for skewed distributions
Q1, Q3 = np.percentile(consumption, [25, 75])
IQR = Q3 - Q1
anomalies = (consumption < Q1 - 1.5*IQR) | (consumption > Q3 + 1.5*IQR)
```

### 3. **Time-Series Specific Methods**

- **EWMA (Exponentially Weighted Moving Average)**: Detect gradual drift
- **Seasonal Decomposition**: Separate trend, seasonal, and residual components
- **Prophet**: Facebook's library for time series with strong seasonal patterns

### 4. **Machine Learning Approaches**

```python
# Isolation Forest - Best for high-dimensional data
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1)
anomalies = iso_forest.fit_predict(features) == -1

# DBSCAN - Groups similar patterns
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=0.5).fit_predict(features)
anomalies = clusters == -1  # Noise points

# Autoencoders - Deep learning approach
# Train to reconstruct normal patterns
# High reconstruction error = anomaly
```

### 5. **Contextual Anomalies for Mosques**

Key anomalies to detect:
- **High night consumption** (2-4 AM should be minimal)
- **Missing prayer peaks** (no spike during prayer times)
- **Constant high baseline** (equipment left on)
- **No Friday increase** (Jummah prayer anomaly)
- **Inverse patterns** (high at night, low during day)

### 6. **Implementation Strategy**

1. **Start Simple**: Use statistical methods first (Z-score, IQR)
2. **Add Context**: Layer in prayer time and day-of-week rules
3. **Machine Learning**: Use Isolation Forest for complex patterns
4. **Combine Methods**: Ensemble approach for robust detection
5. **Validate**: Check detected anomalies against known issues

### 7. **Key Metrics to Track**

- Peak-to-baseline ratio during prayers
- Night-to-day consumption ratio
- Friday-to-weekday ratio
- Seasonal adjustment factors
- Anomaly persistence (consecutive anomalous hours)

### 8. **Alert Thresholds**

- **Critical**: Night consumption > 50% of day average
- **High**: Missing prayer peaks for >2 consecutive prayers
- **Medium**: Baseline drift >20% over a week
- **Low**: Single hour spikes outside prayer times

## Code Implementation

The `mosque_anomaly_detection.py` script implements all these approaches:

```python
# Example usage
from mosque_anomaly_detection import run_mosque_anomaly_detection

df_results, summary = run_mosque_anomaly_detection(
    file_path='your_data.parquet',
    datetime_col='timestamp',
    consumption_col='power',
    meter_col='meter_id'  # Optional
)
```

The script will:
1. Detect statistical anomalies (Z-score, IQR)
2. Find contextual anomalies (prayer times, night consumption)
3. Apply ML-based detection (Isolation Forest)
4. Generate comprehensive visualizations
5. Export anomalous records for investigation