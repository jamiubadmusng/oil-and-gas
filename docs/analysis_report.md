# Oil & Gas Production Analysis Report

## Equinor Volve Field: Production Forecasting and Anomaly Detection

**Author:** Jamiu Olamilekan Badmus  
**Date:** February 2026  
**Industry:** Oil & Gas / Energy

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Source and Acquisition](#data-source-and-acquisition)
3. [About the Volve Field](#about-the-volve-field)
4. [Methodology](#methodology)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Production Forecasting](#production-forecasting)
7. [Anomaly Detection](#anomaly-detection)
8. [Results and Discussion](#results-and-discussion)
9. [Business Recommendations](#business-recommendations)
10. [Conclusions](#conclusions)
11. [References](#references)

---

## 1. Executive Summary

This analysis examines production data from the Equinor Volve oil field in the Norwegian North Sea, covering the period from 2008 to 2016. The project develops machine learning models for:

- **Production forecasting**: Predicting future oil, gas, and water production volumes
- **Anomaly detection**: Identifying unusual patterns that may indicate operational issues

The work demonstrates the application of data science techniques to optimize oil field operations and enable predictive maintenance strategies.

---

## 2. Data Source and Acquisition

### 2.1 Original Data Source

The data originates from **Equinor's Volve Field Open Dataset**, released in June 2018. This historic release made available approximately 40,000 files covering the entire lifecycle of the Volve offshore oil field, including:

- Production logs and reports
- Well sensor data (pressure, temperature, flow rates)
- Drilling records
- Seismic surveys
- Reservoir models

The complete dataset spans several terabytes and represents the most comprehensive data release ever from the Norwegian Continental Shelf.

**Official Source**: [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing)

### 2.2 Data Retrieved For This Project

For this analysis, a curated subset containing daily and monthly production data was obtained from Kaggle:

| Attribute | Details |
|-----------|---------|
| **Dataset Name** | Volve Production Data |
| **Kaggle URL** | https://www.kaggle.com/datasets/lamyalbert/volve-production-data |
| **File Format** | Microsoft Excel (.xlsx) |
| **File Size** | 2.34 MB |
| **Download Date** | February 5, 2026 |
| **License** | Equinor Open Data License |

### 2.3 Data Contents

The Excel file contains two worksheets:

**Sheet 1: Daily Production Data**
- 15,634 records
- 24 columns
- Daily measurements for each wellbore including oil, gas, water volumes, pressures, and temperatures

**Sheet 2: Monthly Production Data**
- 527 records
- 10 columns
- Aggregated monthly production figures per wellbore

### 2.4 How to Obtain the Data

1. Navigate to https://www.kaggle.com/datasets/lamyalbert/volve-production-data
2. Sign in with a Kaggle account (free registration available)
3. Click the "Download" button
4. Extract the Excel file to the project's `data/raw/` folder

---

## 3. About the Volve Field

### 3.1 Field Overview

The Volve field was an offshore oil and gas field located in the southern part of the Norwegian sector of the North Sea, approximately 200 km west of Stavanger, Norway.

| Attribute | Details |
|-----------|---------|
| **Location** | North Sea, Block 15/9 |
| **Water Depth** | 80 meters |
| **Discovery Year** | 1993 |
| **Production Start** | February 2008 |
| **Production End** | September 2016 |
| **Operator** | Equinor (formerly Statoil) |
| **Total Oil Produced** | ~10.7 million Sm³ |
| **Total Gas Produced** | ~2.3 billion Sm³ |

### 3.2 Wells in the Dataset

The dataset includes production data from 7 wellbores:

| Well Name | Type | Description |
|-----------|------|-------------|
| 15/9-F-1 C | Producer | Main production well |
| 15/9-F-4 | Producer | Production well |
| 15/9-F-5 | Producer | Production well |
| 15/9-F-11 | Producer | Production well |
| 15/9-F-12 | Producer | Production well |
| 15/9-F-14 | Producer | Production well |
| 15/9-F-15 D | Injector | Water injection well |

### 3.3 Why Volve Data is Significant

The release of Volve data was groundbreaking for the oil & gas industry:

1. **Completeness**: First time a full field dataset was made publicly available
2. **Real-world complexity**: Contains actual operational challenges and anomalies
3. **Educational value**: Enables training and research without proprietary restrictions
4. **Benchmark dataset**: Allows comparison of analytical techniques across researchers

---

## 4. Methodology

### 4.1 Data Processing Pipeline

```
Raw Excel Data → Data Cleaning → Feature Engineering → Model Training → Evaluation
```

### 4.2 Data Cleaning Steps

1. **Date Parsing**: Convert date columns to proper datetime format
2. **Missing Values**: Identify and handle missing measurements
3. **Outlier Detection**: Flag extreme values for review
4. **Data Type Conversion**: Ensure numeric columns are properly typed
5. **Well Filtering**: Focus on production wells (exclude injection-only periods)

### 4.3 Feature Engineering

**Temporal Features:**
- Day of week, month, quarter, year
- Days since production start
- Time since last maintenance/shutdown

**Rolling Statistics:**
- 7-day, 14-day, 30-day moving averages
- Rolling standard deviations
- Trend indicators (increasing/decreasing)

**Production Metrics:**
- Water cut: Water / (Oil + Water)
- Gas-Oil Ratio (GOR): Gas / Oil
- Decline rate: Day-over-day production change
- Cumulative production

**Operational Features:**
- Choke size changes
- Pressure differentials
- Temperature variations
- On-stream hours percentage

### 4.4 Modeling Approaches

**Time Series Forecasting:**
- ARIMA/SARIMA for univariate forecasting
- Prophet for trend and seasonality decomposition
- LSTM neural networks for sequence learning
- XGBoost with lag features for tabular approach

**Anomaly Detection:**
- Isolation Forest for unsupervised detection
- Statistical methods (Z-score, IQR)
- LSTM Autoencoder for reconstruction-based detection
- Deviation from rolling baselines

---

## 5. Exploratory Data Analysis

### 5.1 Production Overview

The analysis examined production data from the Equinor Volve field spanning approximately 8+ years:

- **Total production period**: September 2007 to December 2016
- **Total daily records**: 15,634 (8,011 active production days after filtering)
- **Wells analyzed**: 6 producer wells (F-1C, F-5, F-11, F-12, F-14, F-15D)
- **Production types**: Oil, Gas, Water

### 5.2 Key Statistics

| Metric | Oil (Sm³) | Gas (Sm³) | Water (Sm³) |
|--------|-----------|-----------|-------------|
| Total | 10.04 million | 1.48 billion | 15.32 million |
| Mean Daily | 1,253 | 184,169 | 1,912 |
| Max Daily | ~9,500 | ~1.3 million | ~25,000 |
| Std Dev | 1,464 | 210,858 | 2,330 |

**Key Derived Metrics:**
- **Water Cut**: Mean 60%, indicating mature field characteristics
- **Gas-Oil Ratio (GOR)**: Mean 151.9 Sm³/Sm³, typical for North Sea reservoirs
- **On-Stream Efficiency**: Average 95.2% when producing

### 5.3 Production Trends

The analysis revealed classic production decline patterns:

1. **Peak Production (2008-2010)**: Field reached peak output with ~8,000-9,000 Sm³/day
2. **Decline Phase (2010-2013)**: Steady exponential decline following reservoir depletion
3. **Mature Phase (2013-2016)**: Stabilized at lower production rates with increased water cut
4. **Seasonal Effects**: Identified annual seasonality pattern (52-week cycle) in decomposition analysis

---

## 6. Production Forecasting

### 6.1 Model Selection

Five regression models were evaluated for production forecasting using time-series features:

| Model | MAPE (%) | RMSE | MAE | R² Score |
|-------|----------|------|-----|----------|
| **Ridge Regression** | **0.61** | **2.22** | **1.95** | **0.9999** |
| Random Forest | 66.65 | 183.90 | 166.80 | 0.19 |
| Gradient Boosting | 67.37 | 183.53 | 163.12 | 0.19 |
| XGBoost | 77.73 | 211.79 | 182.36 | -0.08 |
| LightGBM | 77.64 | 217.85 | 204.67 | -0.14 |

### 6.2 Best Model Performance

**Ridge Regression** achieved exceptional performance with:
- **R² Score**: 99.99% (near-perfect fit)
- **MAPE**: 0.61% (less than 1% average error)
- **MAE**: 1.95 Sm³ (average absolute error)

The outstanding performance is attributed to:
1. High autocorrelation in production time series (lag features capture this well)
2. Linear regularization preventing overfitting on smooth production curves
3. Well-engineered features including rolling means, lags, and EMAs

**Feature Engineering** (24 features total):
- Lag features: 1, 3, 7, 14, 30 days
- Rolling statistics: 7, 14, 30-day windows (mean, std, min, max)
- Differencing: 1-day and 7-day changes
- Exponential moving averages: 7-day and 30-day
- Temporal: day of week, month, quarter
- Operational: on-stream hours, choke size, wellhead pressure, water cut, GOR

### 6.3 Forecast Visualization

See `docs/figures/forecast_results.png` for actual vs predicted production curves showing excellent alignment.

---

## 7. Anomaly Detection

### 7.1 Anomaly Detection Results

Multiple anomaly detection methods were applied:

| Method | Anomalies Detected | Rate (%) | Description |
|--------|-------------------|----------|-------------|
| **Isolation Forest** | **149 days** | **5.03%** | ML-based outlier detection |
| Z-Score (±3σ) | 0 days | 0% | Statistical threshold |
| IQR Method | 282 days | 9.5% | Interquartile range |
| Rolling Baseline | 82 days | 2.8% | Deviation from trend |
| Consensus (≥2 methods) | 0 days | 0% | Multi-method agreement |

### 7.2 Types of Anomalies Identified

The Isolation Forest detected anomalies primarily during:

1. **Peak Production Periods (2009)**: Days with unusually high output (>8,000 Sm³/day)
   - August 2009: 9,436, 9,272, 9,260 Sm³/day flagged
   - December 2009: 8,400+ Sm³/day cluster

2. **Production Transitions**: Periods of rapid change between operational states

3. **Well Startup/Shutdown Events**: Anomalous patterns during operational changes

4. **Equipment Events**: Periods likely associated with maintenance or intervention

**Sample Anomalies Detected:**
| Date | Oil Production (Sm³) | Gas (Sm³) | Water (Sm³) |
|------|---------------------|-----------|-------------|
| 2009-08-10 | 9,436 | 1,301,581 | 993 |
| 2009-08-25 | 9,273 | 1,288,677 | 1,015 |
| 2009-12-09 | 8,422 | 1,175,279 | 2,217 |

### 7.3 Anomaly Visualization

See `docs/figures/anomaly_detection.png` for flagged anomalies on time series plots.

---

## 8. Results and Discussion

### 8.1 Key Findings

1. **Production Decline Patterns**: The Volve field exhibited classic exponential decline behavior, transitioning from peak production of ~9,000 Sm³/day in 2009 to ~1,500 Sm³/day by 2016. This 83% decline over 7 years follows typical mature field characteristics.

2. **Well Performance Variations**: 
   - **F-12**: Top performer with 4.6 million Sm³ total oil (46% of field production)
   - **F-14**: Second performer with 3.9 million Sm³ (39%)
   - **F-11**: Third with 1.2 million Sm³ (12%)
   - Late additions F-1C, F-15D contributed smaller volumes

3. **Seasonal Effects**: Time series decomposition revealed consistent annual seasonality with amplitude of ±500-1,500 Sm³/day, likely reflecting operational schedules and weather impacts on offshore operations.

4. **Operational Correlations**: Strong correlations identified:
   - Oil-Gas: 1.00 (perfect correlation)
   - Wellhead Pressure-Oil: 0.64
   - Water Cut-Oil: -0.55 (negative, as expected for maturing field)

### 8.2 Model Insights

- **Linear models outperformed tree-based models** due to the smooth, autocorrelated nature of production time series
- **Lag features (especially 1-day lag) were most predictive**, confirming strong day-to-day persistence
- **Rolling averages helped capture medium-term trends**, improving predictions during transitional periods

### 8.3 Limitations

1. Historical data only (no real-time predictions without additional infrastructure)
2. Limited to production data (sensor logs would improve anomaly detection)
3. Single field analysis (patterns may differ for other reservoir types)
4. Anomaly labels not available (evaluation limited to unsupervised methods)

---

## 9. Business Recommendations

Based on the analysis, the following recommendations are proposed for oil & gas operations:

### 9.1 Production Optimization

1. **Implement Daily Forecasting**: Deploy the Ridge Regression model for daily production forecasts with sub-1% error, enabling precise operational planning

2. **Focus on Top Performers**: Prioritize F-12 and F-14 for optimization efforts as they contribute 85% of field production

3. **Monitor Water Cut Trends**: Implement alerts when water cut exceeds 70% to trigger reservoir management review

4. **Optimize On-Stream Time**: Target 96%+ on-stream efficiency (current 95.2%) through improved maintenance scheduling

### 9.2 Predictive Maintenance

1. **Alert System**: Configure automated alerts when production deviates >5% from forecast for >3 consecutive days

2. **Pressure Monitoring**: Track wellhead pressure trends as leading indicator of equipment issues (0.64 correlation with production)

3. **Scheduled Interventions**: Plan well interventions during seasonally low production periods (identified in decomposition)

4. **ROI Calculation**: With industry downtime costs of $5M per 3.65-day outage, detecting issues 1 day early could save $1.4M per incident

### 9.3 Anomaly Response Protocols

1. **Tier 1 (Immediate)**: Isolation Forest anomalies during critical production periods → Engineering review within 4 hours

2. **Tier 2 (Standard)**: Statistical anomalies (IQR method) → Review in daily operations meeting

3. **Tier 3 (Monitoring)**: Rolling baseline deviations → Log and track for pattern recognition

---

## 10. Conclusions

This project demonstrates the application of machine learning techniques to real-world oil field production data. Key accomplishments include:

1. **Production Forecasting Model**: Developed a Ridge Regression model achieving 99.99% R² and 0.61% MAPE, enabling highly accurate daily production predictions

2. **Anomaly Detection System**: Implemented Isolation Forest detecting 149 anomalous days (5.03%), successfully identifying peak production events and operational transitions

3. **Production Pattern Analysis**: Identified classic decline curve behavior with 83% production reduction over 7 years, plus annual seasonality patterns

4. **Feature Engineering Framework**: Created 24 production-relevant features including lags, rolling statistics, and operational parameters

5. **Actionable Business Insights**: Quantified potential savings of $1.4M per detected incident through early intervention

**Model Performance Summary:**
| Task | Best Model | Key Metric |
|------|------------|------------|
| Production Forecasting | Ridge Regression | R² = 99.99% |
| Anomaly Detection | Isolation Forest | 149 anomalies (5.03%) |

The techniques demonstrated can be applied to other oil & gas fields to optimize production and reduce unplanned downtime, with potential industry-wide impact of millions of dollars in prevented losses.

---

## 11. References

1. **Equinor (2018)**. "Disclosing all Volve data." Press Release. Available at: https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data

2. **Equinor Volve Data Portal**. https://www.equinor.com/energy/volve-data-sharing

3. **Kaggle Dataset**: Lamy, A. "Volve Production Data." https://www.kaggle.com/datasets/lamyalbert/volve-production-data

4. **SPE (2023)**. "How Predictive Maintenance Is Cutting Back on Costs and Injuries for the Oil and Gas Industry." Journal of Petroleum Technology. https://jpt.spe.org/how-predictive-maintenance-is-cutting-back-on-costs-and-injuries-for-the-oil-and-gas-industry

5. **Energies Media (2024)**. "AI in Oil and Gas: Preventing Equipment Failures Before They Cost Millions." https://energiesmedia.com/ai-in-oil-and-gas-preventing-equipment-failures-before-they-cost-millions/

6. **Energistics**. "Equinor's Volve Field Test Data." https://energistics.org/equinors-volve-field-test-data

---

*Report generated as part of the Oil & Gas Production Analysis project.*

*Contact: jamiubadmus001@gmail.com*
