import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate MAPE while safely handling zero actual values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.where(actual == 0, np.nan, actual)
    return float(np.nanmean(np.abs((actual - predicted) / denominator)) * 100)

try:
    '''
    Why XGBoost? Business data rarely has linear patterns, Gradient Boosting captures complex interactions
    It has built-in feature importance. It automatically tells you what matters, which is critical for business interpretation
    Since this is project is for SMEs whcih usually have few data points, Gradient boosting works well even with limited data. Neural networks would overfit badly here
    It also trains in seconds, so users don't have to wait
    '''
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("Using XGBoost for forecasting")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using sklearn GradientBoostingRegressor as fallback")

@dataclass
class ForecastConfig:
    # stores all the user settings in one place

    target_column: str
    feature_columns: List[str] = field(default_factory=list)
    test_size: float = 0.2
    n_lags: int = 3
    forecast_periods: int = 6
    use_time_features: bool = True
    scale_features: bool = True
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'random_state': 42
    })

    @property
    def xgb_params(self):
        """Backward compatibility property"""
        params = self.model_params.copy()
        if XGBOOST_AVAILABLE:
            params['objective'] = 'reg:squarederror'
            params['colsample_bytree'] = 0.8
        return params
    
@dataclass
class ForecastResult:
    # packages all output together
    predictions: pd.DataFrame
    future_predictions: pd.DataFrame
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model: Any
    config: ForecastConfig
    training_history: Dict[str, List[float]] = field(default_factory=dict)

class TimeSeriesForecaster:

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.training_data: Optional[pd.DataFrame] = None
        self.frequency_delta: Optional[pd.Timedelta] = None

    def prepare_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:

        df_prep = df.copy()

        actual_date_col = None

        if date_column in df_prep.columns:
            actual_date_col = date_column
        
        else:
            for col in df_prep.columns:
                if col.lower() == date_column.lower():
                    actual_date_col = col
                    logger.info(f"Found date column: '{col}' (case-insensitive match)")
                    break

        if actual_date_col is None:
            common_date_names = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 'timestamp', 'Timestamp', 'time', 'Time']

            for col in df_prep.columns:
                if col in common_date_names:
                    actual_date_col = col
                    logger.info(f"Auto-detected date column: '{col}'")
                    break
            
        if actual_date_col is not None:
            df_prep[actual_date_col] = pd.to_datetime(df_prep[actual_date_col])
            df_prep = df_prep.set_index(actual_date_col)
            logger.info(f"Set '{actual_date_col}' as datetime index")
        
        else:
            raise ValueError(f"Could not find date column. Available columns: {list(df_prep.columns)}")

        df_prep = df_prep.sort_index()
        df_prep = df_prep.groupby(df_prep.index).agg(
            lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0]
        )

        # Convert any remaining datetime columns into numeric calendar features.
        # Raw datetime dtypes cannot be passed directly to sklearn/XGBoost feature matrices.
        datetime_cols = df_prep.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns.tolist()
        for col in datetime_cols:
            df_prep[f'{col}_year'] = df_prep[col].dt.year
            df_prep[f'{col}_month'] = df_prep[col].dt.month
            df_prep[f'{col}_day'] = df_prep[col].dt.day
            df_prep[f'{col}_dayofweek'] = df_prep[col].dt.dayofweek
        if datetime_cols:
            df_prep = df_prep.drop(columns=datetime_cols)

        target = self.config.target_column

        for lag in range(1, self.config.n_lags + 1):
            df_prep[f'{target}_lag_{lag}'] = df_prep[target].shift(lag)
        
        # Create rolling statistics
        if len(df_prep) >= 3:
            df_prep[f'{target}_rolling_mean_3'] = df_prep[target].rolling(window=3, min_periods=1).mean()
            df_prep[f'{target}_rolling_std_3'] = df_prep[target].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Add time-based features
        # This tell the model when this is happening. Without the time features the model just sees it as random numbers
        if self.config.use_time_features:
            df_prep['month'] = df_prep.index.month
            df_prep['quarter'] = df_prep.index.quarter
            df_prep['year'] = df_prep.index.year
            df_prep['day_of_year'] = df_prep.index.dayofyear
            
            # Encodes months as a circle, not a line. Enf of year patterns continue into the new year
            df_prep['month_sin'] = np.sin(2 * np.pi * df_prep['month'] / 12)
            df_prep['month_cos'] = np.cos(2 * np.pi * df_prep['month'] / 12)
        
        # Add lag features for user-provided business drivers
        if self.config.feature_columns:
            for feature in self.config.feature_columns:
                if feature in df_prep.columns and feature != target:
                    df_prep[f'{feature}_lag_1'] = df_prep[feature].shift(1)
        
        df_prep = df_prep.dropna()

        categorical_cols = df_prep.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self.config.target_column]

        if len(categorical_cols) > 0:
            df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)
        
        logger.info(f"Created {len(df_prep.columns)} features from {len(df.columns)} original columns")
        
        return df_prep
        
    def train(self, df: pd.DataFrame, date_column: str = 'date') -> ForecastResult:
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        df_features = self.prepare_features(df, date_column)

        if len(df_features) < 10:
            raise ValueError(f"Insufficient data after feature engineering. Need at least 10 rows, got {len(df_features)}")
        
        target = self.config.target_column

        feature_cols = [col for col in df_features.columns 
                       if col != target and not col.startswith(f'{target}_')]
        
        lag_cols = [col for col in df_features.columns 
                   if col.startswith(f'{target}_lag_') or col.startswith(f'{target}_rolling_')]
        feature_cols.extend(lag_cols)
        
        self.feature_names = feature_cols
        
        X = df_features[feature_cols]
        y = df_features[target]

        self.training_data = df_features.copy()
        self.frequency_delta = self._infer_time_delta(df_features.index)
        '''
        This is Time-Series Split not random split. It simulates real forecsting-predict future based on past
        bet for time series forecasting
        '''
        test_size = int(len(X) * self.config.test_size)

        if test_size < 2:
            test_size = 2

        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]

        # For Scaling Features
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        logger.info(f"Training {'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'} model...")

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**self.config.xgb_params)

            eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        else:
            # Use sklearn GradientBoostingRegressor as fallback
            sklearn_params = {
                'max_depth': self.config.model_params.get('max_depth', 4),
                'learning_rate': self.config.model_params.get('learning_rate', 0.1),
                'n_estimators': self.config.model_params.get('n_estimators', 100),
                'subsample': self.config.model_params.get('subsample', 0.8),
                'random_state': self.config.model_params.get('random_state', 42)
            }
            self.model = GradientBoostingRegressor(**sklearn_params)
            self.model.fit(X_train_scaled, y_train)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mape': _safe_mape(y_train, y_train_pred),
            'test_mape': _safe_mape(y_test, y_test_pred)
        }
        
        logger.info(f"Model trained. Test RMSE: {metrics['test_rmse']:.2f}, Test R²: {metrics['test_r2']:.3f}")
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'date': df_features.index,
            'actual': y,
            'predicted': np.concatenate([y_train_pred, y_test_pred]),
            'split': ['train'] * len(y_train) + ['test'] * len(y_test)
        })

        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        feature_importance = {k: float(v) for k, v in sorted(feature_importance.items(), 
                                                              key=lambda x: x[1], 
                                                              reverse=True)}
        
        # Generate future predictions
        future_predictions = self.generate_future_predictions(df_features)
        
        # Get training history
        if XGBOOST_AVAILABLE and hasattr(self.model, 'evals_result'):
            training_history = {
                'train_rmse': self.model.evals_result()['validation_0']['rmse'],
                'test_rmse': self.model.evals_result()['validation_1']['rmse']
            }
        else:
            # For sklearn, compute manually
            training_history = {
                'train_rmse': [metrics['train_rmse']],
                'test_rmse': [metrics['test_rmse']]
            }
        
        result = ForecastResult(
            predictions=predictions_df,
            future_predictions=future_predictions,
            metrics=metrics,
            feature_importance=feature_importance,
            model=self.model,
            config=self.config,
            training_history=training_history
        )
        
        return result

    def _infer_time_delta(self, date_index: pd.Index) -> pd.Timedelta:
        """Infer a reasonable step between forecast periods."""
        if len(date_index) <= 1:
            return pd.Timedelta(days=30)

        freq = pd.infer_freq(date_index)
        if freq is not None:
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                last_date = date_index[-1]
                return (last_date + offset) - last_date
            except Exception:
                pass

        date_diffs = date_index[1:] - date_index[:-1]
        return pd.Timedelta(np.median([d.total_seconds() for d in date_diffs]), unit='s')

    def _build_future_feature_row(
        self,
        next_date: pd.Timestamp,
        target_history: List[float],
        exogenous_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Construct a future feature row from lagged history and stable driver assumptions."""
        exogenous_values = exogenous_values or {}
        feature_row = pd.DataFrame(index=[next_date])
        target = self.config.target_column

        if self.config.use_time_features:
            feature_row['month'] = next_date.month
            feature_row['quarter'] = next_date.quarter
            feature_row['year'] = next_date.year
            feature_row['day_of_year'] = next_date.dayofyear
            feature_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
            feature_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)

        for lag in range(1, self.config.n_lags + 1):
            history_idx = -lag
            lag_value = target_history[history_idx] if len(target_history) >= lag else target_history[0]
            feature_row[f'{target}_lag_{lag}'] = lag_value

        recent_window = target_history[-3:] if len(target_history) >= 3 else target_history
        feature_row[f'{target}_rolling_mean_3'] = float(np.mean(recent_window))
        feature_row[f'{target}_rolling_std_3'] = float(np.std(recent_window)) if len(recent_window) > 1 else 0.0

        last_known = self.training_data.iloc[-1] if self.training_data is not None else pd.Series(dtype=float)

        for col in self.feature_names:
            if col in feature_row.columns:
                continue

            if col in exogenous_values:
                feature_row[col] = exogenous_values[col]
            elif col.endswith('_lag_1'):
                base_feature = col[:-6]
                feature_row[col] = exogenous_values.get(
                    base_feature,
                    last_known.get(base_feature, last_known.get(col, 0.0))
                )
            elif col in last_known.index:
                feature_row[col] = last_known[col]
            else:
                feature_row[col] = 0.0

        return feature_row[self.feature_names]

    def _predict_feature_row(self, feature_row: pd.DataFrame) -> float:
        """Generate a prediction from a prepared feature row."""
        if self.model is None:
            raise ValueError("Model must be trained before generating predictions")

        if self.scaler is not None:
            feature_scaled = self.scaler.transform(feature_row)
        else:
            feature_scaled = feature_row.values

        return float(self.model.predict(feature_scaled)[0])
    
    def generate_future_predictions(self, df_features: pd.DataFrame) -> pd.DataFrame:

        if self.model is None:
            raise ValueError("Model must be trained before generating predictions")
        
        future_dates = []
        future_values = []
        last_date = df_features.index[-1]
        step_delta = self.frequency_delta or self._infer_time_delta(df_features.index)
        target_history = df_features[self.config.target_column].tolist()
        exogenous_values = {
            feature: float(df_features[feature].iloc[-1])
            for feature in self.config.feature_columns
            if feature in df_features.columns
        }
        
        for step in range(1, self.config.forecast_periods + 1):
            next_date = last_date + step_delta * step
            feature_row = self._build_future_feature_row(next_date, target_history, exogenous_values)
            pred_value = self._predict_feature_row(feature_row)
            future_dates.append(next_date)
            future_values.append(pred_value)
            target_history.append(pred_value)
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted': future_values,
            'type': 'forecast'
        })
        
        return future_df
    
    def predict(self, df: pd.DataFrame, date_column: str = 'date') -> pd.Series:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with features
            date_column: Name of date column
            
        Returns:
            Series of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        df_features = self.prepare_features(df, date_column)
        X = df_features[self.feature_names]
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=df_features.index)
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get feature importance as a DataFrame"""
        if self.model is None:
            return pd.DataFrame()
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        
        df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        return df


def calculate_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive forecasting metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
        'mae': mean_absolute_error(actual, predicted),
        'r2': r2_score(actual, predicted),
        'mape': _safe_mape(actual, predicted),
        'max_error': np.max(np.abs(actual - predicted)),
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted)
    }
    
    return metrics


def create_forecast_summary(result: ForecastResult) -> Dict[str, Any]:
    """
    Create a human-readable summary of forecast results
    
    Args:
        result: ForecastResult object
        
    Returns:
        Dictionary with summary information
    """
    summary = {
        'target_variable': result.config.target_column,
        'training_samples': len(result.predictions[result.predictions['split'] == 'train']),
        'test_samples': len(result.predictions[result.predictions['split'] == 'test']),
        'forecast_periods': len(result.future_predictions),
        'forecast_mean': float(result.future_predictions['predicted'].mean()),
        'forecast_min': float(result.future_predictions['predicted'].min()),
        'forecast_max': float(result.future_predictions['predicted'].max()),
        'last_actual': float(result.predictions['actual'].iloc[-1]),
        'average_forecast_change_pct': float(
            ((result.future_predictions['predicted'].mean() - result.predictions['actual'].iloc[-1])
             / result.predictions['actual'].iloc[-1]) * 100
        ) if result.predictions['actual'].iloc[-1] != 0 else 0.0,
        'model_performance': {
            'test_rmse': f"{result.metrics['test_rmse']:.2f}",
            'test_mae': f"{result.metrics['test_mae']:.2f}",
            'test_r2': f"{result.metrics['test_r2']:.3f}",
            'test_mape': f"{result.metrics['test_mape']:.2f}%"
        },
        'top_features': list(result.feature_importance.keys())[:5],
        'forecast_range': {
            'start': result.future_predictions['date'].iloc[0].strftime('%Y-%m-%d'),
            'end': result.future_predictions['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    }
    
    return summary

    



