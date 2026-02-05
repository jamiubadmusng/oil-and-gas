# Oil & Gas: Production Forecasting and Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## Predicting Oil Well Production and Detecting Operational Anomalies

**Author:** Jamiu Olamilekan Badmus  
📧 [jamiubadmus001@gmail.com](mailto:jamiubadmus001@gmail.com)  
🔗 [GitHub](https://github.com/jamiubadmusng) | [LinkedIn](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/) | [Website](https://sites.google.com/view/jamiu-olamilekan-badmus/)

---

## 📋 Executive Summary

This project analyzes production data from the **Equinor Volve oil field** (2008-2016) in the Norwegian North Sea to build predictive models for:

1. **Production Forecasting**: Time-series models to predict future oil, gas, and water production
2. **Anomaly Detection**: Identifying abnormal operational patterns that may indicate equipment issues or production problems

### Business Impact

In the oil & gas industry, unplanned downtime is extremely costly:
- Upstream companies face **~27 days of downtime per year** costing **$38 million**
- A single **3.65-day outage** can cost over **$5 million**
- Predictive maintenance using ML can **save hundreds of thousands of dollars per hour** of prevented downtime

### Key Results

| Metric | Value |
|--------|-------|
| Wells Analyzed | 5 producers + 2 injectors |
| Production Records | 15,634 daily records |
| Time Period | 2008-2016 (8+ years) |
| Total Oil Production | 10.04 million Sm³ |
| Total Gas Production | 1.48 billion Sm³ |
| Best Forecasting Model | **Ridge Regression** |
| Forecasting R² Score | **99.99%** |
| Forecasting MAPE | **0.61%** |
| Anomalies Detected | **149 days (5.03%)** |

---

## 🎯 Project Objectives

1. **Analyze Production Trends**: Examine historical oil, gas, and water production patterns across all wells
2. **Build Forecasting Models**: Develop time-series models to predict future production volumes
3. **Detect Anomalies**: Identify unusual production patterns that may indicate operational issues
4. **Feature Engineering**: Create domain-relevant features (decline rates, rolling statistics, etc.)
5. **Provide Actionable Insights**: Deliver recommendations for production optimization and maintenance planning

---

## 📊 Dataset

### Source: Equinor Volve Field Open Dataset

The Volve oil field was an offshore field in the Norwegian North Sea, operated by Equinor (formerly Statoil) from 2008 to 2016. In June 2018, Equinor made history by releasing the complete dataset from the field for research and education purposes—the most comprehensive data release ever from the Norwegian Continental Shelf.

The full Volve dataset comprises approximately **40,000 files** including production logs, well sensor data, drilling records, seismic surveys, and reservoir models (several terabytes total).

### Data Retrieved From

**Kaggle Dataset**: [Volve Production Data](https://www.kaggle.com/datasets/lamyalbert/volve-production-data)

This curated subset contains the essential daily and monthly production data extracted from the original Equinor release, making it ideal for time-series analysis and machine learning applications.

**Download Date**: February 5, 2026

**Download Instructions**:
1. Visit https://www.kaggle.com/datasets/lamyalbert/volve-production-data
2. Sign in to Kaggle (create free account if needed)
3. Click "Download" button
4. Extract the Excel file to `data/raw/` folder

### File Details

| File | Description | Size | Records |
|------|-------------|------|---------|
| `Volve production data.xlsx` | Daily & Monthly production data | ~2.34 MB | 15,634 daily + 527 monthly |

### Data Dictionary

#### Daily Production Data (Sheet 1)

| Column | Description | Unit |
|--------|-------------|------|
| DATEPRD | Production date | Date |
| WELL_BORE_CODE | Well identifier | String |
| NPD_WELL_BORE_CODE | Norwegian Petroleum Directorate code | String |
| NPD_WELL_BORE_NAME | Official well name | String |
| NPD_FIELD_CODE | Field identifier | Integer |
| NPD_FIELD_NAME | Field name (VOLVE) | String |
| NPD_FACILITY_CODE | Facility code | Integer |
| NPD_FACILITY_NAME | Facility name | String |
| ON_STREAM_HRS | Hours producing | Hours |
| AVG_DOWNHOLE_PRESSURE | Average downhole pressure | Bar |
| AVG_DOWNHOLE_TEMPERATURE | Average downhole temperature | °C |
| AVG_DP_TUBING | Average differential pressure | Bar |
| AVG_ANNULUS_PRESS | Average annulus pressure | Bar |
| AVG_CHOKE_SIZE_P | Average choke size | mm |
| AVG_WHP_P | Average wellhead pressure | Bar |
| AVG_WHT_P | Average wellhead temperature | °C |
| DP_CHOKE_SIZE | Choke size for DP | mm |
| BORE_OIL_VOL | Oil produced | Sm³ |
| BORE_GAS_VOL | Gas produced | Sm³ |
| BORE_WAT_VOL | Water produced | Sm³ |
| BORE_WI_VOL | Water injected | Sm³ |
| FLOW_KIND | Flow type (production/injection) | String |
| WELL_TYPE | Well type (producer/injector) | String |

#### Monthly Production Data (Sheet 2)

| Column | Description | Unit |
|--------|-------------|------|
| Year | Production year | Integer |
| Month | Production month | Integer |
| Wellbore | Well identifier | String |
| Oil (Sm3) | Monthly oil production | Sm³ |
| Gas (Sm3) | Monthly gas production | Sm³ |
| Water (Sm3) | Monthly water production | Sm³ |
| Water Injection (Sm3) | Monthly water injection | Sm³ |
| Days on Production | Days the well was active | Days |
| Gas Injection (Sm3) | Monthly gas injection | Sm³ |
| CO2 Injection (1000Sm3) | CO2 injection volume | 1000 Sm³ |

### License

This data is provided under the **Equinor Open Data License**, which permits use for academic research, study, and development purposes.

---

## 🏗️ Project Structure

```
oil-and-gas/
├── README.md                          # Project overview (this file)
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Volve production data.xlsx
│   └── processed/                    # Cleaned and transformed data
│       └── production_data_processed.csv
│
├── docs/
│   ├── analysis_report.md            # Detailed analysis write-up
│   └── figures/                      # Generated visualizations
│       ├── production_overview.png
│       ├── well_comparison.png
│       ├── time_series_decomposition.png
│       ├── forecast_results.png
│       ├── anomaly_detection.png
│       └── feature_importance.png
│
├── models/
│   ├── production_forecaster.joblib  # Trained forecasting model
│   ├── anomaly_detector.joblib       # Trained anomaly detection model
│   └── scaler.joblib                 # Data scaler
│
├── notebooks/
│   └── oil_gas_production_analysis.ipynb  # Main analysis notebook
│
└── src/
    ├── __init__.py
    └── predict_production.py         # Production prediction module
```

---

## 🔬 Methodology

### 1. Data Preparation
- Load Excel data with daily and monthly production records
- Parse dates and handle missing values
- Filter for production wells (exclude injection-only periods)
- Calculate derived metrics (water cut, GOR, decline rates)

### 2. Exploratory Data Analysis
- Production trends over the field lifetime
- Individual well performance comparison
- Seasonal and cyclical patterns
- Correlation between operational parameters and production

### 3. Feature Engineering
- **Rolling Statistics**: 7-day, 30-day moving averages
- **Lag Features**: Previous day/week/month production
- **Decline Rates**: Production decline over time
- **Operational Features**: Choke size, pressure differentials
- **Water Cut**: Water/(Oil+Water) ratio
- **Gas-Oil Ratio (GOR)**: Gas/Oil production ratio

### 4. Production Forecasting

| Model | Description |
|-------|-------------|
| ARIMA/SARIMA | Classical time-series with seasonality |
| Prophet | Facebook's forecasting library |
| LSTM | Deep learning sequence model |
| XGBoost | Gradient boosting with lag features |
| Random Forest | Ensemble method |

### 5. Anomaly Detection

| Method | Description |
|--------|-------------|
| Isolation Forest | Tree-based anomaly detection |
| Statistical Thresholds | Z-score and IQR methods |
| LSTM Autoencoder | Deep learning reconstruction error |
| Moving Average Deviation | Deviation from rolling baseline |

### 6. Model Evaluation

**Forecasting Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

**Anomaly Detection Metrics:**
- Precision, Recall, F1-Score
- ROC-AUC (if labeled data available)
- Visual validation by domain experts

---

## 📈 Key Results

### Production Overview
- **Total oil produced**: 10.04 million Sm³
- **Total gas produced**: 1.48 billion Sm³
- **Total water produced**: 15.32 million Sm³
- **Average daily oil production**: 1,253 Sm³/day
- **Field production period**: 2008-2016 (8+ years)

### Forecasting Performance
- **Best performing model**: Ridge Regression
- **R² Score**: 99.99%
- **MAPE on test set**: 0.61%
- **MAE**: 1.95 Sm³

### Anomaly Detection
- **Total anomalies detected**: 149 days (5.03%)
- **Detection method**: Isolation Forest
- **Peak anomalies**: 2009 (highest production period)

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/jamiubadmusng/oil-and-gas.git
cd oil-and-gas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lamyalbert/volve-production-data)
2. Place `Volve production data.xlsx` in the `data/raw/` folder

### Running the Analysis

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/oil_gas_production_analysis.ipynb
```

### Using the Prediction Module

```python
from src.predict_production import ProductionForecaster

# Initialize forecaster
forecaster = ProductionForecaster()
forecaster.load_models('models/')

# Predict next 30 days for a well
predictions = forecaster.forecast(
    well_id='15/9-F-12',
    horizon=30
)

# Detect anomalies
anomalies = forecaster.detect_anomalies(
    well_id='15/9-F-12',
    data=recent_data
)
```

---

## 🔑 Key Insights

### 1. Production Decline Analysis
The Volve field exhibited classic exponential decline behavior, transitioning from peak production of ~9,000 Sm³/day in 2009 to ~1,500 Sm³/day by 2016—an 83% decline over 7 years following typical mature field characteristics.

### 2. Well Performance Comparison
- **F-12**: Top performer with 4.6 million Sm³ total oil (46% of field production)
- **F-14**: Second performer with 3.9 million Sm³ (39%)
- **F-11**: Third with 1.2 million Sm³ (12%)
- Late additions F-1C and F-15D contributed smaller volumes

### 3. Operational Anomalies
- Isolation Forest detected 149 anomalous days (5.03% of production days)
- Peak anomalies concentrated during 2009 (highest production period)
- Anomalies correspond to extreme production values and operational transitions

### 4. Recommendations
- Deploy Ridge Regression model for daily production forecasts (sub-1% error)
- Prioritize F-12 and F-14 for optimization efforts (85% of production)
- Configure alerts when production deviates >5% from forecast for >3 days
- Plan well interventions during seasonally low production periods

---

## 🛠️ Technologies Used

- **Python 3.10+**: Core programming language
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: Machine learning models
- **XGBoost/LightGBM**: Gradient boosting
- **Prophet**: Time-series forecasting
- **Statsmodels**: ARIMA and statistical models
- **TensorFlow/Keras**: LSTM models
- **Joblib**: Model serialization

---

## 📚 References

1. [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing) - Official dataset release
2. [Kaggle: Volve Production Data](https://www.kaggle.com/datasets/lamyalbert/volve-production-data) - Dataset source
3. [How Predictive Maintenance Is Cutting Back on Costs](https://jpt.spe.org/how-predictive-maintenance-is-cutting-back-on-costs-and-injuries-for-the-oil-and-gas-industry) - SPE Journal
4. [AI in Oil and Gas](https://energiesmedia.com/ai-in-oil-and-gas-preventing-equipment-failures-before-they-cost-millions/) - Industry applications

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The underlying Volve dataset is provided under the [Equinor Open Data License](https://cdn.equinor.com/files/h61q9gi9/global/de6532f6134b9a953f6c41bac47a0c055a3712d3.pdf).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

**Jamiu Olamilekan Badmus**

- 📧 Email: [jamiubadmus001@gmail.com](mailto:jamiubadmus001@gmail.com)
- 🔗 GitHub: [jamiubadmusng](https://github.com/jamiubadmusng)
- 💼 LinkedIn: [Jamiu Olamilekan Badmus](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- 🌐 Website: [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)
