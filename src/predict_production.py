"""
Oil & Gas Production Forecasting and Anomaly Detection Module

This module provides classes for:
1. Loading and preprocessing Volve production data
2. Forecasting oil, gas, and water production
3. Detecting operational anomalies

Author: Jamiu Olamilekan Badmus
Email: jamiubadmus001@gmail.com
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib


class VolveDataLoader:
    """
    Loader class for Volve production data from Excel file.
    
    Attributes:
        data_path: Path to the Excel file
        daily_data: DataFrame with daily production records
        monthly_data: DataFrame with monthly production records
    """
    
    def __init__(self, data_path: Union[str, Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the Volve production data Excel file
        """
        self.data_path = Path(data_path) if data_path else None
        self.daily_data = None
        self.monthly_data = None
        
    def load_data(self, data_path: Union[str, Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load daily and monthly production data from Excel file.
        
        Args:
            data_path: Path to the Excel file (optional if set in __init__)
            
        Returns:
            Tuple of (daily_data, monthly_data) DataFrames
        """
        if data_path:
            self.data_path = Path(data_path)
            
        if self.data_path is None:
            raise ValueError("Data path must be provided")
            
        print(f"Loading data from: {self.data_path}")
        
        # Load daily production data
        self.daily_data = pd.read_excel(
            self.data_path, 
            sheet_name='Daily Production Data',
            parse_dates=['DATEPRD']
        )
        
        # Load monthly production data
        self.monthly_data = pd.read_excel(
            self.data_path, 
            sheet_name='Monthly Production Data'
        )
        
        print(f"Loaded {len(self.daily_data):,} daily records")
        print(f"Loaded {len(self.monthly_data):,} monthly records")
        
        return self.daily_data, self.monthly_data
    
    def get_well_list(self) -> List[str]:
        """Get list of unique wells in the dataset."""
        if self.daily_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.daily_data['NPD_WELL_BORE_NAME'].unique().tolist()
    
    def get_well_data(self, well_name: str) -> pd.DataFrame:
        """
        Get production data for a specific well.
        
        Args:
            well_name: Name of the well (e.g., '15/9-F-12')
            
        Returns:
            DataFrame with production data for the specified well
        """
        if self.daily_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.daily_data[self.daily_data['NPD_WELL_BORE_NAME'] == well_name].copy()


class ProductionForecaster:
    """
    Production forecasting model for oil, gas, and water volumes.
    
    Supports multiple forecasting approaches:
    - XGBoost/Random Forest with lag features
    - Gradient Boosting
    - Simple baseline models
    """
    
    def __init__(self):
        """Initialize the forecaster."""
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.target_columns = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'BORE_OIL_VOL') -> pd.DataFrame:
        """
        Create time-series features for forecasting.
        
        Args:
            df: DataFrame with production data (must have DATEPRD column)
            target_col: Target column for creating lag features
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = df.sort_values('DATEPRD')
        
        # Temporal features
        df['year'] = df['DATEPRD'].dt.year
        df['month'] = df['DATEPRD'].dt.month
        df['day'] = df['DATEPRD'].dt.day
        df['day_of_week'] = df['DATEPRD'].dt.dayofweek
        df['day_of_year'] = df['DATEPRD'].dt.dayofyear
        df['quarter'] = df['DATEPRD'].dt.quarter
        df['week_of_year'] = df['DATEPRD'].dt.isocalendar().week.astype(int)
        
        # Days since start of production
        df['days_since_start'] = (df['DATEPRD'] - df['DATEPRD'].min()).dt.days
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            
        # Trend features
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_7'] = df[target_col].diff(7)
        
        # Production ratios (if multiple production columns exist)
        if 'BORE_OIL_VOL' in df.columns and 'BORE_WAT_VOL' in df.columns:
            total_liquid = df['BORE_OIL_VOL'] + df['BORE_WAT_VOL']
            df['water_cut'] = np.where(total_liquid > 0, df['BORE_WAT_VOL'] / total_liquid, 0)
            
        if 'BORE_OIL_VOL' in df.columns and 'BORE_GAS_VOL' in df.columns:
            df['gor'] = np.where(df['BORE_OIL_VOL'] > 0, df['BORE_GAS_VOL'] / df['BORE_OIL_VOL'], 0)
        
        # Operational features (if available)
        if 'ON_STREAM_HRS' in df.columns:
            df['on_stream_pct'] = df['ON_STREAM_HRS'] / 24.0
            
        if 'AVG_CHOKE_SIZE_P' in df.columns:
            df['choke_change'] = df['AVG_CHOKE_SIZE_P'].diff(1)
            
        return df
    
    def train(self, df: pd.DataFrame, target_col: str = 'BORE_OIL_VOL', 
              model_type: str = 'gradient_boosting') -> Dict:
        """
        Train a forecasting model.
        
        Args:
            df: DataFrame with production data
            target_col: Column to predict
            model_type: Type of model ('random_forest', 'gradient_boosting')
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features
        df_features = self.prepare_features(df, target_col)
        
        # Remove rows with NaN (from lag features)
        df_features = df_features.dropna()
        
        # Define feature columns
        exclude_cols = ['DATEPRD', 'WELL_BORE_CODE', 'NPD_WELL_BORE_CODE', 
                       'NPD_WELL_BORE_NAME', 'NPD_FIELD_NAME', 'NPD_FACILITY_NAME',
                       'FLOW_KIND', 'WELL_TYPE'] + self.target_columns
        
        self.feature_columns = [col for col in df_features.columns 
                               if col not in exclude_cols and df_features[col].dtype in ['int64', 'float64']]
        
        X = df_features[self.feature_columns]
        y = df_features[target_col]
        
        # Scale features
        self.scalers[target_col] = StandardScaler()
        X_scaled = self.scalers[target_col].fit_transform(X)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:  # gradient_boosting
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
        model.fit(X_scaled, y)
        self.models[target_col] = model
        
        # Calculate training metrics
        y_pred = model.predict(X_scaled)
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': mean_absolute_percentage_error(y[y > 0], y_pred[y > 0]) * 100
        }
        
        print(f"Training complete for {target_col}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, target_col: str = 'BORE_OIL_VOL') -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            df: DataFrame with production data (needs same structure as training data)
            target_col: Column to predict
            
        Returns:
            Array of predictions
        """
        if target_col not in self.models:
            raise ValueError(f"Model for {target_col} not trained. Call train() first.")
            
        df_features = self.prepare_features(df, target_col)
        df_features = df_features.dropna()
        
        X = df_features[self.feature_columns]
        X_scaled = self.scalers[target_col].transform(X)
        
        return self.models[target_col].predict(X_scaled)
    
    def get_feature_importance(self, target_col: str = 'BORE_OIL_VOL') -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            target_col: Target column for which to get feature importance
            
        Returns:
            DataFrame with feature importance scores
        """
        if target_col not in self.models:
            raise ValueError(f"Model for {target_col} not trained.")
            
        importance = self.models[target_col].feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_models(self, path: Union[str, Path]):
        """Save trained models and scalers to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            joblib.dump(model, path / f'forecaster_{target}.joblib')
            
        for target, scaler in self.scalers.items():
            joblib.dump(scaler, path / f'scaler_{target}.joblib')
            
        # Save feature columns
        joblib.dump(self.feature_columns, path / 'feature_columns.joblib')
        
        print(f"Models saved to {path}")
        
    def load_models(self, path: Union[str, Path], target_cols: List[str] = None):
        """Load trained models and scalers from disk."""
        path = Path(path)
        
        if target_cols is None:
            target_cols = self.target_columns
            
        for target in target_cols:
            model_path = path / f'forecaster_{target}.joblib'
            scaler_path = path / f'scaler_{target}.joblib'
            
            if model_path.exists():
                self.models[target] = joblib.load(model_path)
                
            if scaler_path.exists():
                self.scalers[target] = joblib.load(scaler_path)
                
        # Load feature columns
        fc_path = path / 'feature_columns.joblib'
        if fc_path.exists():
            self.feature_columns = joblib.load(fc_path)
            
        print(f"Models loaded from {path}")


class AnomalyDetector:
    """
    Anomaly detection for production data using multiple methods.
    
    Methods supported:
    - Isolation Forest
    - Statistical thresholds (Z-score, IQR)
    - Rolling baseline deviation
    """
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.isolation_forest = None
        self.scaler = None
        self.baseline_stats = {}
        
    def fit_isolation_forest(self, df: pd.DataFrame, 
                            feature_cols: List[str] = None,
                            contamination: float = 0.05) -> 'AnomalyDetector':
        """
        Fit Isolation Forest model for anomaly detection.
        
        Args:
            df: DataFrame with production data
            feature_cols: Columns to use for detection (defaults to production volumes)
            contamination: Expected proportion of anomalies
            
        Returns:
            Self for method chaining
        """
        if feature_cols is None:
            feature_cols = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'ON_STREAM_HRS']
            feature_cols = [col for col in feature_cols if col in df.columns]
            
        X = df[feature_cols].fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        self.feature_cols = feature_cols
        print(f"Isolation Forest fitted on {len(X)} samples")
        
        return self
    
    def detect_isolation_forest(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect anomalies using fitted Isolation Forest.
        
        Args:
            df: DataFrame with production data
            
        Returns:
            Series with anomaly labels (-1 for anomaly, 1 for normal)
        """
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not fitted. Call fit_isolation_forest() first.")
            
        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return pd.Series(self.isolation_forest.predict(X_scaled), index=df.index)
    
    def detect_statistical(self, df: pd.DataFrame, 
                          col: str = 'BORE_OIL_VOL',
                          method: str = 'zscore',
                          threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalies using statistical methods.
        
        Args:
            df: DataFrame with production data
            col: Column to analyze
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for anomaly detection
            
        Returns:
            Series with boolean anomaly flags
        """
        values = df[col].fillna(0)
        
        if method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            return z_scores > threshold
            
        elif method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (values < lower_bound) | (values > upper_bound)
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_rolling_deviation(self, df: pd.DataFrame,
                                 col: str = 'BORE_OIL_VOL',
                                 window: int = 30,
                                 threshold: float = 2.0) -> pd.Series:
        """
        Detect anomalies based on deviation from rolling baseline.
        
        Args:
            df: DataFrame with production data
            col: Column to analyze
            window: Rolling window size in days
            threshold: Number of standard deviations for anomaly threshold
            
        Returns:
            Series with boolean anomaly flags
        """
        values = df[col].fillna(0)
        
        rolling_mean = values.rolling(window=window, center=True).mean()
        rolling_std = values.rolling(window=window, center=True).std()
        
        deviation = np.abs(values - rolling_mean)
        return deviation > (threshold * rolling_std)
    
    def detect_all(self, df: pd.DataFrame, col: str = 'BORE_OIL_VOL') -> pd.DataFrame:
        """
        Run all anomaly detection methods and return combined results.
        
        Args:
            df: DataFrame with production data
            col: Column to analyze
            
        Returns:
            DataFrame with anomaly flags from each method
        """
        results = pd.DataFrame(index=df.index)
        
        # Isolation Forest (if fitted)
        if self.isolation_forest is not None:
            if_labels = self.detect_isolation_forest(df)
            results['isolation_forest'] = if_labels == -1
            
        # Statistical methods
        results['zscore'] = self.detect_statistical(df, col, method='zscore', threshold=3.0)
        results['iqr'] = self.detect_statistical(df, col, method='iqr', threshold=1.5)
        
        # Rolling deviation
        results['rolling_deviation'] = self.detect_rolling_deviation(df, col, window=30, threshold=2.0)
        
        # Consensus (anomaly if flagged by multiple methods)
        results['consensus'] = results.sum(axis=1) >= 2
        
        return results
    
    def save(self, path: Union[str, Path]):
        """Save fitted models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.isolation_forest is not None:
            joblib.dump(self.isolation_forest, path / 'isolation_forest.joblib')
            joblib.dump(self.scaler, path / 'anomaly_scaler.joblib')
            joblib.dump(self.feature_cols, path / 'anomaly_features.joblib')
            
        print(f"Anomaly detector saved to {path}")
        
    def load(self, path: Union[str, Path]):
        """Load fitted models from disk."""
        path = Path(path)
        
        if (path / 'isolation_forest.joblib').exists():
            self.isolation_forest = joblib.load(path / 'isolation_forest.joblib')
            self.scaler = joblib.load(path / 'anomaly_scaler.joblib')
            self.feature_cols = joblib.load(path / 'anomaly_features.joblib')
            
        print(f"Anomaly detector loaded from {path}")


def main():
    """
    Main function demonstrating the production analysis pipeline.
    """
    # Example usage
    print("=" * 60)
    print("Oil & Gas Production Analysis Module")
    print("=" * 60)
    
    # Define paths
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'Volve production data.xlsx'
    
    if not data_path.exists():
        print(f"\nData file not found at: {data_path}")
        print("Please download from: https://www.kaggle.com/datasets/lamyalbert/volve-production-data")
        return
        
    # Load data
    print("\n1. Loading Data...")
    loader = VolveDataLoader(data_path)
    daily_data, monthly_data = loader.load_data()
    
    # Show wells
    print("\n2. Wells in Dataset:")
    for well in loader.get_well_list():
        print(f"   - {well}")
        
    # Train forecaster on first production well
    print("\n3. Training Forecaster...")
    forecaster = ProductionForecaster()
    
    # Get a production well's data
    wells = loader.get_well_list()
    well_data = loader.get_well_data(wells[0])
    
    if len(well_data) > 100:  # Need enough data for training
        metrics = forecaster.train(well_data, target_col='BORE_OIL_VOL')
        
        # Feature importance
        print("\n4. Top 10 Important Features:")
        importance = forecaster.get_feature_importance()
        print(importance.head(10).to_string(index=False))
    
    # Anomaly detection
    print("\n5. Fitting Anomaly Detector...")
    detector = AnomalyDetector()
    detector.fit_isolation_forest(daily_data)
    
    anomalies = detector.detect_all(daily_data)
    print(f"   Total anomalies detected (consensus): {anomalies['consensus'].sum()}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
