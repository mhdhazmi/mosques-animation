# Anomalous Nighttime Consumption Analysis Report

## Overview
Analysis of electrical load profile data to identify meters with anomalous consumption patterns during nighttime hours (9 PM - 4 AM).

## Dataset Information
- **Total Records Analyzed**: 500,000 (subset of 1.5M+ total records)
- **Date Range**: May 30, 2022 to May 20, 2023
- **Unique Meters**: 31 in sample
- **Anomalous Meters Found**: 8

## Anomaly Detection Criteria
1. **High nighttime consumption** compared to peak (>70% of overall maximum)
2. **Night consumption higher than day** consumption average
3. **High base consumption** during night (doesn't dip close to minimum)
4. **Increasing consumption trend** during nighttime hours

## Top Anomalous Meters

### 1. Meter AES2020896472402 (Highest Anomaly Score: 4)
**Primary Issues:**
- Night average (982W) is **2.2x higher** than day average (453W)
- High night minimum (896W) vs overall mean (629W)
- **Critical Finding**: Consumption increases at night instead of decreasing

**Potential Causes:**
- Inappropriate usage (facilities running at night)
- Faulty equipment or heating/cooling systems
- Possible unauthorized usage
- Industrial processes running during off-peak hours

### 2. Meter KFM2020660044515 (Anomaly Score: 4)
**Primary Issues:**
- Night average (12W) slightly higher than day average (11W)
- High baseline consumption that doesn't vary significantly

### 3. Meter KFM2020660037773 (Anomaly Score: 2)
**Primary Issues:**
- High night minimum (36W) compared to overall mean (66W)
- Consumption doesn't drop to expected low levels during night

## Key Findings

### Consumption Patterns
- **Normal Pattern**: Consumption should typically decrease during 9 PM - 4 AM
- **Anomalous Pattern**: Consumption remains high or increases during nighttime

### Statistical Analysis
- **Night vs Day Ratio**: Anomalous meters show ratios >1.0 (normal should be <0.7)
- **Base Load Analysis**: Night minimum consumption should be <30% of daily average
- **Peak Shift**: Some meters show peak consumption during traditionally low-demand hours

## Recommendations

### Immediate Actions
1. **Investigate Meter AES2020896472402** - highest priority due to 2.2x night consumption
2. **On-site inspection** of anomalous meters for equipment issues
3. **Usage pattern analysis** to identify unauthorized or inappropriate usage

### Long-term Monitoring
1. **Real-time alerts** for consumption patterns outside normal ranges
2. **Monthly reporting** on night-to-day consumption ratios
3. **Trend analysis** to identify emerging anomalies

### Technical Considerations
1. **Equipment malfunction** detection protocols
2. **Load balancing** optimization for identified anomalous meters
3. **Energy efficiency** audits for high base-load consumers

## Generated Visualizations

1. **anomalous_consumption_analysis.png** - Overview of top 3 anomalous meters with hourly patterns and time series
2. **consumption_heatmap_anomalous_meters.png** - Heatmap showing hourly consumption patterns across anomalous meters
3. **detailed_analysis_AES2020896472402.png** - Comprehensive analysis of the most anomalous meter
4. **top_anomalous_meters_comparison.png** - Side-by-side comparison of top 3 anomalous meters

## Data Files
- **anomalous_meters_analysis.csv** - Detailed statistics for all anomalous meters
- **combined_load_profile_electrical.csv** - Cleaned and combined dataset

---
*Analysis completed on: May 29, 2025*
*Total processing time: Analysis of 500K+ records*