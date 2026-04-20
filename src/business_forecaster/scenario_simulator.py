import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScenarioConfig:
    """Configuration fro scenario simulation"""
    name: str # e.g, "Optimistic", "Pessimistic", "Base Case"
    adjustments: Dict[str, float] # e.g., {'marketing_spend': 1.20} = +20%
    description: str = " "

    def apply_to_value(self, feature_name: str, original_value: float) -> float:
        """
        Apply adjustment to a feature value

        Args:
            feature_name: Name of the feature
            original_value: Original value
            
        Returns:
            Adjusted value
        """

        if feature_name in self.adjustments:
            multiplier = self.adjustments[feature_name]
            return original_value * multiplier
        return original_value
    
@dataclass
class ScenarioResult:
    """Results from a scenario simulation"""
    scenario_name: str
    config: ScenarioConfig
    predictions: pd.DataFrame # Future predictions with adjusted inputs
    base_predictions: pd.DataFrame # Original predictions for comparison
    impact_summary: Dict[str, float] # Summary of changes
    adjusted_drivers: Dict[str, float] = field(default_factory=dict)

    def get_average_change(self) -> float:
        """Calculate average % change from base case"""
        base_avg = self.base_predictions['predicted'].mean()
        scenario_avg = self.predictions['predicted'].mean()
        return ((scenario_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0.0
    
class ScenarioSimulator:
    """
    Enables what-if analysis by adjusting input features and re-running predictions
    """

    def __init__(self, forecaster, base_result):
        """
        Initialize scenario simulator
        
        Args:
            forecaster: Trained TimeSeriesForecaster instance
            base_result: ForecastResult from the base (original) forecast
        """
        self.forecaster = forecaster
        self.base_result = base_result
        self.scenarios: Dict[str, ScenarioResult] = {}

    def create_scenario(self, name:str, adjustments: Dict[str, float], description: str = " ") -> ScenarioConfig:
        """
        Create a scenario configuration
        
        Args:
            name: Scenario name (e.g., "High Growth")
            adjustments: Dict of {feature_name: multiplier}
                        e.g., {'marketing_spend': 1.20} means +20%
                        e.g., {'marketing_spend': 0.80} means -20%
            description: Human-readable description
            
        Returns:
            ScenarioConfig object
        """
        config = ScenarioConfig(
            name=name,
            adjustments=adjustments,
            description=description
        )
        return config
    
    def run_scenario(self, config: ScenarioConfig) -> ScenarioResult:
        """
        Run a scenario simulation
        
        Args:
            config: ScenarioConfig with adjustments
            
        Returns:
            ScenarioResult with predictions and comparisons
        """
        logger.info(f"Running scenario: {config.name}")

        if self.forecaster.training_data is None or self.forecaster.model is None:
            raise ValueError("Forecaster must be trained before running scenario analysis")

        adjusted_drivers = self._build_adjusted_driver_values(config)

        # Generate predictions with adjusted features
        predictions = self._generate_adjusted_predictions(adjusted_drivers, config.name)

        #calculate impact summary
        impact_summary = self._calculate_impact(predictions, self.base_result.future_predictions)

        # Create result
        result = ScenarioResult(
            scenario_name=config.name,
            config=config,
            predictions=predictions,
            base_predictions=self.base_result.future_predictions,
            impact_summary=impact_summary,
            adjusted_drivers=adjusted_drivers
        )
        
        # Store for later comparison
        self.scenarios[config.name] = result

        logger.info(f"Scenario '{config.name}' complete. Average change: {result.get_average_change():.2f}%")
        return result
    
    def _build_adjusted_driver_values(self, config: ScenarioConfig) -> Dict[str, float]:
        """Apply scenario multipliers to user-controlled business drivers."""
        last_known = self.forecaster.training_data.iloc[-1]
        adjusted_drivers: Dict[str, float] = {}

        for feature_name in self.forecaster.config.feature_columns:
            if feature_name not in last_known.index:
                continue

            original_value = float(last_known[feature_name])
            adjusted_drivers[feature_name] = config.apply_to_value(feature_name, original_value)

        return adjusted_drivers
    
    def _generate_adjusted_predictions(self,
                                       adjusted_drivers: Dict[str, float],
                                       scenario_name: str) -> pd.DataFrame:
        """
        Generate predictions with adjusted feature values

        Args:
            adjusted_drivers: Dict of {feature_name: adjusted_value}

        Returns:
            DataFrame with scenario predictions
        """
        future_dates = []
        future_values = []
        last_date = self.forecaster.training_data.index[-1]
        date_diff = self.forecaster.frequency_delta or pd.Timedelta(days=30)
        target_history = self.forecaster.training_data[self.forecaster.config.target_column].tolist()

        # Generate predictions for each future period
        for step in range(1, self.forecaster.config.forecast_periods + 1):
            next_date = last_date + (date_diff * step)
            feature_vector = self.forecaster._build_future_feature_row(
                next_date,
                target_history,
                adjusted_drivers
            )
            pred_value = self.forecaster._predict_feature_row(feature_vector)

            future_dates.append(next_date)
            future_values.append(pred_value)
            target_history.append(pred_value)

        # Create result DataFrame
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted': future_values,
            'scenario': scenario_name
        })

        return predictions_df
    
    def _calculate_impact(self,
                          scenario_predictions: pd.DataFrame,
                          base_predictions: pd.DataFrame) -> Dict[str, float]:
        
        """
        Calculate impact metrics comparing scenario to base case

        Args:
        scenario_predictions: Predictions from scenario
        base_predictions: Original base predictions

        Returns:
        Dict with impact metrics
        """
        scenario_avg = scenario_predictions['predicted'].mean()
        base_avg = base_predictions['predicted'].mean()

        scenario_total = scenario_predictions['predicted'].sum()
        base_total = base_predictions['predicted'].sum()
        peak_change = scenario_predictions['predicted'].max() - base_predictions['predicted'].max()
        trough_change = scenario_predictions['predicted'].min() - base_predictions['predicted'].min()

        avg_change_pct = ((scenario_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0.0
        total_change_pct = ((scenario_total - base_total) / base_total) * 100 if base_total != 0 else 0.0

        impact = {
            'base_average': base_avg,
            'scenario_average': scenario_avg,
            'average_change': scenario_avg - base_avg,
            'average_change_pct': avg_change_pct,
            'base_total': base_total,
            'scenario_total': scenario_total,
            'total_change': scenario_total - base_total,
            'total_change_pct': total_change_pct,
            'peak_change': peak_change,
            'trough_change': trough_change
        }

        return impact
    
    def compare_scenarios(self, scenario_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple scenarios

        Args:
            scenario_names: List of scenario names to compare (None = all)

        Returns:
            DataFrame with comparison metrics
        """

        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())

        comparison_data = []

        # Add base case
        base_avg = self.base_result.future_predictions['predicted'].mean()
        comparison_data.append({
            'Scenario': 'Base Case',
            'Average Prediction': base_avg,
            'Change from Base': 0,
            'Change %': 0
        })

        # Add each scenario
        for name in scenario_names:
            if name in self.scenarios:
                result = self.scenarios[name]
                scenario_avg = result.predictions['predicted'].mean()
                change = scenario_avg - base_avg
                change_pct = (change / base_avg) * 100

                comparison_data.append({
                    'Scenario': name,
                    'Average Prediction': scenario_avg,
                    'Change from Base': change,
                    'Change %': change_pct
                })

        return pd.DataFrame(comparison_data)
    
def create_preset_scenarios(feature_columns: List[str]) -> List[ScenarioConfig]:
        """
        Create preset scenario configurations

        Args:
            feature_columns: List of available feature columns

        Returns:
            List of preset ScenarioConfig objects
        """

        presets = []

        # Optimistic scenario: Increase all controllable features by 10%
        if len(feature_columns) > 0:
            optimistic_adjustments = {col: 1.10 for col in feature_columns}
            presets.append(ScenarioConfig(
                name="Optimistic (+10%)",
                adjustments=optimistic_adjustments,
                description="All input features increased by 10%"
            ))

        # Pessimistic scenario: Decrease all by 10%
        if len(feature_columns) > 0:
            pessimistic_adjustments = {col: 0.90 for col in feature_columns}
            presets.append(ScenarioConfig(
                name="Pessimistic (-10%)",
                adjustments=pessimistic_adjustments,
                description="All input features decreased by 10%"
            ))
        
        # Conservative scenario: Decrease by 20%
        if len(feature_columns) > 0:
            conservative_adjustments = {col: 0.80 for col in feature_columns}
            presets.append(ScenarioConfig(
                name="Conservative (-20%)",
                adjustments=conservative_adjustments,
                description="All input features decreased by 20%"
            ))
        
        # Aggressive scenario: Increase by 25%
        if len(feature_columns) > 0:
            aggressive_adjustments = {col: 1.25 for col in feature_columns}
            presets.append(ScenarioConfig(
                name="Aggressive (+25%)",
                adjustments=aggressive_adjustments,
                description="All input features increased by 25%"
            ))

        return presets
    
