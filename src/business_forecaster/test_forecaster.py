import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from business_forecaster.forecaster import (
    TimeSeriesForecaster, 
    ForecastConfig, 
    calculate_forecast_metrics,
    create_forecast_summary
)


class TestForecastConfig(unittest.TestCase):
    """Test ForecastConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ForecastConfig(target_column='revenue')
        
        self.assertEqual(config.target_column, 'revenue')
        self.assertEqual(config.test_size, 0.2)
        self.assertEqual(config.n_lags, 3)
        self.assertEqual(config.forecast_periods, 6)
        self.assertTrue(config.use_time_features)
        self.assertTrue(config.scale_features)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ForecastConfig(
            target_column='sales',
            test_size=0.3,
            n_lags=5,
            forecast_periods=12
        )
        
        self.assertEqual(config.target_column, 'sales')
        self.assertEqual(config.test_size, 0.3)
        self.assertEqual(config.n_lags, 5)
        self.assertEqual(config.forecast_periods, 12)


class TestTimeSeriesForecaster(unittest.TestCase):
    """Test TimeSeriesForecaster class"""
    
    def setUp(self):
        """Create sample data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='ME')
        
        # Generate synthetic data with trend and seasonality
        trend = np.linspace(1000, 2000, 50)
        seasonality = 100 * np.sin(np.linspace(0, 4*np.pi, 50))
        noise = np.random.normal(0, 50, 50)
        
        revenue = trend + seasonality + noise
        
        self.df = pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'marketing_spend': revenue * 0.1 + np.random.normal(0, 10, 50),
            'customer_count': (revenue / 10) + np.random.normal(0, 5, 50)
        })
        
        self.config = ForecastConfig(
            target_column='revenue',
            feature_columns=['marketing_spend', 'customer_count'],
            test_size=0.2,
            n_lags=3,
            forecast_periods=6
        )
    
    def test_initialization(self):
        """Test forecaster initialization"""
        forecaster = TimeSeriesForecaster(self.config)
        
        self.assertIsNotNone(forecaster)
        self.assertEqual(forecaster.config.target_column, 'revenue')
        self.assertIsNone(forecaster.model)
    
    def test_prepare_features(self):
        """Test feature engineering"""
        forecaster = TimeSeriesForecaster(self.config)
        df_features = forecaster.prepare_features(self.df, 'date')
        
        # Check lag features were created
        self.assertIn('revenue_lag_1', df_features.columns)
        self.assertIn('revenue_lag_2', df_features.columns)
        self.assertIn('revenue_lag_3', df_features.columns)
        
        # Check rolling features
        self.assertIn('revenue_rolling_mean_3', df_features.columns)
        
        # Check time features
        self.assertIn('month', df_features.columns)
        self.assertIn('quarter', df_features.columns)
        
        # Check that rows with NaN were dropped
        self.assertEqual(df_features.isnull().sum().sum(), 0)
    
    def test_train_model(self):
        """Test model training"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        # Check that model was trained
        self.assertIsNotNone(forecaster.model)
        
        # Check result structure
        self.assertIsNotNone(result.predictions)
        self.assertIsNotNone(result.future_predictions)
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.feature_importance)
        
        # Check metrics are reasonable
        self.assertGreater(result.metrics['test_r2'], -1)
        self.assertLess(result.metrics['test_r2'], 1.1)
        self.assertGreater(result.metrics['test_rmse'], 0)
    
    def test_predictions_structure(self):
        """Test prediction DataFrame structure"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        # Check predictions DataFrame
        self.assertIn('date', result.predictions.columns)
        self.assertIn('actual', result.predictions.columns)
        self.assertIn('predicted', result.predictions.columns)
        self.assertIn('split', result.predictions.columns)
        
        # Check train/test split
        train_count = (result.predictions['split'] == 'train').sum()
        test_count = (result.predictions['split'] == 'test').sum()
        
        self.assertGreater(train_count, 0)
        self.assertGreater(test_count, 0)
    
    def test_future_predictions(self):
        """Test future prediction generation"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        # Check future predictions
        self.assertEqual(len(result.future_predictions), self.config.forecast_periods)
        self.assertIn('date', result.future_predictions.columns)
        self.assertIn('predicted', result.future_predictions.columns)
        
        # Check dates are in the future
        last_train_date = result.predictions['date'].max()
        first_future_date = result.future_predictions['date'].min()
        
        self.assertGreater(first_future_date, last_train_date)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        # Check feature importance
        self.assertGreater(len(result.feature_importance), 0)
        
        # Check that importance values are valid
        for feature, importance in result.feature_importance.items():
            self.assertGreaterEqual(importance, 0)
            self.assertIsInstance(feature, str)
    
    def test_insufficient_data(self):
        """Test error handling with insufficient data"""
        small_df = self.df.head(5)  # Too few rows
        forecaster = TimeSeriesForecaster(self.config)
        
        with self.assertRaises(ValueError):
            forecaster.train(small_df, 'date')
    
    def test_missing_target_column(self):
        """Test error handling with missing target"""
        df_missing = self.df.drop('revenue', axis=1)
        forecaster = TimeSeriesForecaster(self.config)
        
        with self.assertRaises(ValueError):
            forecaster.train(df_missing, 'date')
    
    def test_predict_new_data(self):
        """Test making predictions on new data"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        # Create new data for prediction
        new_dates = pd.date_range(start='2027-07-01', periods=5, freq='ME')
        new_df = pd.DataFrame({
            'date': new_dates,
            'revenue': [1500, 1550, 1600, 1650, 1700],
            'marketing_spend': [150, 155, 160, 165, 170],
            'customer_count': [150, 155, 160, 165, 170]
        })
        
        predictions = forecaster.predict(new_df, 'date')
        
        self.assertEqual(len(predictions), len(new_df) - self.config.n_lags)
        self.assertTrue(all(predictions > 0))
    
    def test_feature_importance_summary(self):
        """Test feature importance summary DataFrame"""
        forecaster = TimeSeriesForecaster(self.config)
        result = forecaster.train(self.df, 'date')
        
        importance_df = forecaster.get_feature_importance_summary()
        
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Check sorted by importance
        self.assertTrue(importance_df['importance'].is_monotonic_decreasing)


class TestForecastMetrics(unittest.TestCase):
    """Test forecast metric calculations"""
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([102, 108, 122, 128, 142])
        
        metrics = calculate_forecast_metrics(actual, predicted)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Check values are reasonable
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertLess(metrics['mape'], 100)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = actual.copy()
        
        metrics = calculate_forecast_metrics(actual, predicted)
        
        self.assertAlmostEqual(metrics['rmse'], 0, places=5)
        self.assertAlmostEqual(metrics['mae'], 0, places=5)
        self.assertAlmostEqual(metrics['r2'], 1.0, places=5)
        self.assertAlmostEqual(metrics['mape'], 0, places=5)


class TestForecastSummary(unittest.TestCase):
    """Test forecast summary creation"""
    
    def setUp(self):
        """Create sample data and train model"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='ME')
        revenue = np.linspace(1000, 2000, 30) + np.random.normal(0, 50, 30)
        
        df = pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'marketing_spend': revenue * 0.1
        })
        
        config = ForecastConfig(
            target_column='revenue',
            feature_columns=['marketing_spend'],
            forecast_periods=6
        )
        
        forecaster = TimeSeriesForecaster(config)
        self.result = forecaster.train(df, 'date')
    
    def test_summary_structure(self):
        """Test summary dictionary structure"""
        summary = create_forecast_summary(self.result)
        
        self.assertIn('target_variable', summary)
        self.assertIn('training_samples', summary)
        self.assertIn('test_samples', summary)
        self.assertIn('forecast_periods', summary)
        self.assertIn('model_performance', summary)
        self.assertIn('top_features', summary)
        self.assertIn('forecast_range', summary)
    
    def test_summary_values(self):
        """Test summary values are correct"""
        summary = create_forecast_summary(self.result)
        
        self.assertEqual(summary['target_variable'], 'revenue')
        self.assertEqual(summary['forecast_periods'], 6)
        self.assertGreater(summary['training_samples'], 0)
        self.assertGreater(summary['test_samples'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_minimal_data(self):
        """Test with minimal viable data"""
        dates = pd.date_range(start='2023-01-01', periods=15, freq='ME')
        df = pd.DataFrame({
            'date': dates,
            'revenue': np.linspace(1000, 1500, 15)
        })
        
        config = ForecastConfig(
            target_column='revenue',
            n_lags=2,
            forecast_periods=3
        )
        
        forecaster = TimeSeriesForecaster(config)
        result = forecaster.train(df, 'date')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.future_predictions), 3)
    
    def test_single_feature(self):
        """Test with only target variable (no additional features)"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='ME')
        df = pd.DataFrame({
            'date': dates,
            'revenue': np.linspace(1000, 2000, 30) + np.random.normal(0, 50, 30)
        })
        
        config = ForecastConfig(
            target_column='revenue',
            feature_columns=[]  # No additional features
        )
        
        forecaster = TimeSeriesForecaster(config)
        result = forecaster.train(df, 'date')
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.feature_importance), 0)
    
    def test_no_time_features(self):
        """Test with time features disabled"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='ME')
        df = pd.DataFrame({
            'date': dates,
            'revenue': np.linspace(1000, 2000, 30)
        })
        
        config = ForecastConfig(
            target_column='revenue',
            use_time_features=False
        )
        
        forecaster = TimeSeriesForecaster(config)
        result = forecaster.train(df, 'date')
        
        self.assertIsNotNone(result)
    
    def test_no_scaling(self):
        """Test with feature scaling disabled"""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='ME')
        df = pd.DataFrame({
            'date': dates,
            'revenue': np.linspace(1000, 2000, 30)
        })
        
        config = ForecastConfig(
            target_column='revenue',
            scale_features=False
        )
        
        forecaster = TimeSeriesForecaster(config)
        result = forecaster.train(df, 'date')
        
        self.assertIsNotNone(result)
        self.assertIsNone(forecaster.scaler)


if __name__ == '__main__':
    unittest.main()