import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dataloader import DataLoader, DataPreprocessor, DataValidator
from forecaster import TimeSeriesForecaster, ForecastConfig, create_forecast_summary


# Title
st.set_page_config(
    page_title="Business Performance Forecaster",
    page_icon="📊",
    layout="wide"
)

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = None
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None


st.title("Dynamic Business Performance Forecaster")
st.markdown("Upload your business data to start forecasting and scenario planning")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Dataset", "📋 Data Summary", "🔮 Forecasting", "Manual Entry"])

# --- Feature 1: Load and Preprocess ---
with tab1:
    st.header("Load Business Dataset")
    st.markdown("Upload your business data (CSV or Excel format) ")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel(.xlsx, .xls)"
    )

    if uploaded_file is not None:
        
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.session_state.current_df = df

            else:
                df = pd.read_excel(uploaded_file)

            st.success(f" Successfully loaded {len(df)} rows")

            # Start Basic preprocessing
            st.subheader("Data Validation")
            validator = DataValidator(min_rows=10)
            is_valid, issues = validator.validate_dataset(df)

            if is_valid:
                st.success("Dataset passes basic validation")

            else:
                st.warning("Validation Issues found")
                for issue in issues:
                    st.write(f"- {issue}")

            if validator.validation_results.get('warnings'):
                with st.expander("⚠️ Data Quality Warnings"):
                    for warning in validator.validation_results['warnings']:
                        st.write(f"- {warning}")

            st.session_state['df'] = df
            st.success(f"Data loaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Data Preview ---
with tab2:
    if 'df' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state['df'].head(10))
        
        st.subheader("Quick Stats")
        st.write(st.session_state['df'].describe())



# with tab3:
#     st.header("Manual Data Entry")
#     st.markdown("Enter business varibales and historical data manially")

#     entry_tab1, entry_tab2 = st.tabs(["Define Variables", "Historical Data"])

#     with entry_tab1:
#         st.subheader("Add Business Variables")

#         with st.expander("Common Business Variables"):
#             common_vars = ManualDataEntry.get_common_variables_list()
#             common_df = pd.DataFrame(common_vars)
#             st.dataframe(common_df, use_container_width=True)

#             with st.form("add_variable_form"):
#                 col1, col2, col3 = st.columns(3)

#                 with col1:
#                     var_name = st.text_input("Variable Name*", placeholder="variable name")
#                     var_value = st.number_input("Current value*", value=0.0)

#                 with col2:
#                     var_unit = st.text_input("Unit", placeholder="e.g., USD, %, units")
#                     var_category = st.selectbox(
#                         "Category", ["general", "revenue", "cost", "marketing", "operations", "external"]
#                     )

#                 with col3:
#                     var_description = st.text_input("Description", placeholder="Brief description")
#                     is_target = st.checkbox("This is the target variable to forecast")

#                 submitted = st.form_submit_button("Add variable")

with tab3:
    st.header("🔮 ML Forecasting")
    st.markdown("Train a machine learning model to forecast business performance")
    
    data_available = False
    data_source = None
    forecast_df = None
    
    if st.session_state.current_df is not None and len(st.session_state.current_df) >= 15:
        data_available = True
        data_source = "uploaded"
        forecast_df = st.session_state.current_df.copy()
    
    
    if not data_available:
        st.warning("⚠️ Insufficient data for forecasting. Please either:")
        st.write("- Upload a dataset with at least 15 rows, OR")
        st.write("- Enter at least 10 historical data points manually")
    else:
        st.success(f"✅ Using {'uploaded dataset' if data_source == 'uploaded' else 'manually entered data'} ({len(forecast_df)} observations)")
        
        # Configuration section
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select target variable
            numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
            
            target_var = st.selectbox(
                "Target Variable (what to forecast)*",
                numeric_cols,
                help="Select the variable you want to predict"
            )
            
            # Select feature columns
            feature_options = [col for col in numeric_cols if col != target_var]
            feature_cols = st.multiselect(
                "Feature Variables (predictors)",
                feature_options,
                default=feature_options[:3] if len(feature_options) >= 3 else feature_options,
                help="Additional variables to use for prediction"
            )
        
        with col2:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Portion of data to use for testing"
            )
            
            forecast_periods = st.number_input(
                "Forecast Periods",
                min_value=1,
                max_value=24,
                value=6,
                help="Number of future periods to predict"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_lags = st.number_input(
                    "Number of Lags",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="How many past periods to use as features"
                )
            
            with col2:
                use_time_features = st.checkbox(
                    "Use Time Features",
                    value=True,
                    help="Add month, quarter, year as features"
                )
            
            with col3:
                scale_features = st.checkbox(
                    "Scale Features",
                    value=True,
                    help="Normalize features before training"
                )
        
        # Train model button
        if st.button("🚀 Train Model & Generate Forecast", type="primary"):
            with st.spinner("Training model... This may take a moment."):
                try:
                    # Create configuration
                    config = ForecastConfig(
                        target_column=target_var,
                        feature_columns=feature_cols,
                        test_size=test_size,
                        n_lags=n_lags,
                        forecast_periods=forecast_periods,
                        use_time_features=use_time_features,
                        scale_features=scale_features
                    )
                    
                    # Initialize and train forecaster
                    forecaster = TimeSeriesForecaster(config)
                    date_column = None
                    possible_date_names = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 
                                        'timestamp', 'Timestamp', 'time', 'Time']

                    for col in forecast_df.columns:
                        if col in possible_date_names:
                            date_column = col
                            st.info(f"📅 Auto-detected date column: '{col}'")
                            break

                    if date_column is None:
                        # Let user select
                        date_column = st.selectbox(
                            "Select Date Column",
                            forecast_df.columns.tolist(),
                            help="Which column contains your dates?"
                        )
                    result = forecaster.train(forecast_df, date_column)
                    
                    # Store result in session state
                    st.session_state.forecast_result = result
                    st.session_state.forecaster = forecaster
                    
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.session_state.forecast_result = None
        
        # Display results if available
        if st.session_state.forecast_result is not None:
            result = st.session_state.forecast_result
            forecaster = st.session_state.get("forecaster", None)
            summary = create_forecast_summary(result)
            latest_actual = summary['last_actual']
            avg_change_pct = summary['average_forecast_change_pct']
            future_df = result.future_predictions.copy()
            
            st.markdown("---")
            st.subheader("📊 Forecast Results")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Test RMSE",
                    f"{result.metrics['test_rmse']:.2f}",
                    help="Root Mean Squared Error on test set"
                )
            
            with col2:
                st.metric(
                    "Test R²",
                    f"{result.metrics['test_r2']:.3f}",
                    help="R-squared score (1.0 = perfect fit)"
                )
            
            with col3:
                st.metric(
                    "Test MAE",
                    f"{result.metrics['test_mae']:.2f}",
                    help="Mean Absolute Error on test set"
                )
            
            with col4:
                st.metric(
                    "Test MAPE",
                    f"{result.metrics['test_mape']:.1f}%",
                    help="Mean Absolute Percentage Error"
                )

            st.markdown("### Forecast Interpretation")

            insight_col1, insight_col2, insight_col3 = st.columns(3)

            with insight_col1:
                st.metric("Latest Actual", f"{latest_actual:,.2f}")

            with insight_col2:
                st.metric(
                    "Average Forecast",
                    f"{summary['forecast_mean']:,.2f}",
                    delta=f"{avg_change_pct:+.1f}% vs latest actual"
                )

            with insight_col3:
                st.metric(
                    "Forecast Range",
                    f"{summary['forecast_max'] - summary['forecast_min']:,.2f}",
                    help="Difference between the highest and lowest projected forecast values"
                )

            performance_message = "strong" if result.metrics['test_r2'] >= 0.7 else "moderate" if result.metrics['test_r2'] >= 0.4 else "limited"
            error_message = "low" if result.metrics['test_mape'] <= 10 else "moderate" if result.metrics['test_mape'] <= 20 else "high"
            top_driver_text = ", ".join(
                feature.replace('_lag_1', ' (previous period)')
                for feature in summary['top_features'][:3]
            )

            st.info(
                f"This model shows {performance_message} fit on the held-out test data. "
                f"The next {result.config.forecast_periods} forecast periods average {summary['forecast_mean']:,.2f}, "
                f"which is {avg_change_pct:+.1f}% compared with the latest observed {summary['target_variable']} value. "
                f"Expected forecast error is {error_message} based on a test MAPE of {result.metrics['test_mape']:.1f}%."
            )

            if top_driver_text:
                st.caption(f"Most influential drivers in this run: {top_driver_text}.")
            
            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Predictions", "Future Forecast", "Feature Importance", "Scenario Analysis"])
            
            with viz_tab1:
                st.markdown("### Historical Predictions vs Actuals")
                
                # Prepare data for plotting
                pred_df = result.predictions.copy()
                
                # Create the plot
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                
                # Add predictions (train)
                train_df = pred_df[pred_df['split'] == 'train']
                fig.add_trace(go.Scatter(
                    x=train_df['date'],
                    y=train_df['predicted'],
                    mode='lines',
                    name='Predicted (Train)',
                    line=dict(color='lightgreen', width=2, dash='dot')
                ))
                
                # Add predictions (test)
                test_df = pred_df[pred_df['split'] == 'test']
                fig.add_trace(go.Scatter(
                    x=test_df['date'],
                    y=test_df['predicted'],
                    mode='lines+markers',
                    name='Predicted (Test)',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{target_var} - Predictions vs Actual",
                    xaxis_title="Date",
                    yaxis_title=target_var,
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show predictions table
                with st.expander("📋 View Predictions Table"):
                    st.dataframe(pred_df, use_container_width=True)
            
            with viz_tab2:
                st.markdown("### Future Forecast")

                # Combine historical actual and future predictions
                fig = go.Figure()
                
                # Historical actual
                fig.add_trace(go.Scatter(
                    x=result.predictions['date'],
                    y=result.predictions['actual'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Future predictions
                fig.add_trace(go.Scatter(
                    x=future_df['date'],
                    y=future_df['predicted'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='orange', width=3, dash='dash'),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"{target_var} - {forecast_periods} Period Forecast",
                    xaxis_title="Date",
                    yaxis_title=target_var,
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.markdown("**Forecast Values:**")
                forecast_display = future_df.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                forecast_display['predicted'] = forecast_display['predicted'].round(2)
                st.dataframe(forecast_display, use_container_width=True)
                st.caption(
                    f"Projected values span from {summary['forecast_min']:,.2f} to {summary['forecast_max']:,.2f} "
                    f"between {summary['forecast_range']['start']} and {summary['forecast_range']['end']}."
                )
            
            with viz_tab3:
                st.markdown("### Feature Importance")
                st.write("Features ranked by their impact on predictions:")
                
                # Get feature importance
                importance_items = list(result.feature_importance.items())[:10]  # Top 10
                features, importances = zip(*importance_items) if importance_items else ([], [])
                
                fig = go.Figure(go.Bar(
                    x=list(importances),
                    y=list(features),
                    orientation='h',
                    marker=dict(
                        color=list(importances),
                        colorscale='Viridis'
                    )
                ))
                
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=500,
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show full table
                with st.expander("📋 View All Feature Importances"):
                    importance_df = pd.DataFrame({
                        'Feature': result.feature_importance.keys(),
                        'Importance': result.feature_importance.values()
                    })
                    st.dataframe(importance_df, use_container_width=True)

            with viz_tab4:
                st.markdown("### Scenario Analysis - Waht-If Simulations")
                st.write("Adjust input variables to see how predictions change")

                # Import scenario simulator
                from scenario_simulator import ScenarioSimulator, create_preset_scenarios
                forecaster = st.session_state.get("forecaster")
                simulator = None

                # Initialize simulator
                if forecaster is None:
                    st.error("Forecaster not available. Please train the model first")

                else:
                    simulator = ScenarioSimulator(forecaster, result)
                

                # Create two columns: controls and results
                control_col, result_col = st.columns([1,2])

                with control_col:
                    st.subheader("scenario Controls")
                    scenario_features = result.config.feature_columns

                    # Preset scenarios
                    st.markdown("**Quick Presets:**")
                    presets = create_preset_scenarios(scenario_features)

                    selected_preset = st.selectbox(
                        "Choose a preset scenario",
                        ["Custom"] + [p.name for p in presets],
                        help="Pre-configured scenarios for quick analysis"
                    )

                    # Custom adjustments
                    st.markdown("**CUstom Adjustments:**")
                    st.caption("Adjust each feature (1.0 = no change, 1.2 = +20%, 0.8 = -20%)")

                    adjustments = {}

                    # If preset selected, use those values as defaults
                    if selected_preset != "Custom":
                        preset_config = next(p for p in presets if p.name == selected_preset)
                        default_adjustments = preset_config.adjustments

                    else:
                        default_adjustments = {col: 1.0 for col in scenario_features}

                    # Create sliders for each feature
                    for feature in scenario_features:
                        default_val = default_adjustments.get(feature, 1.0)

                        adjustment = st.slider(
                            f"{feature}",
                            min_value=0.5,
                            max_value=2.0,
                            value=float(default_val),
                            step=0.05,
                            format="%.2fx",
                            help=f"Multiplier for {feature}, Current: {default_val:.2f}x"
                        )
                        adjustments[feature] = adjustment

                    # Scenario name
                    scenario_name = st.text_input(
                        "Scenario Name",
                        value=selected_preset if selected_preset != "Custom" else "My scenario"
                    )

                    # Run  button
                    run_scenario = st.button("Run Scenario", type="primary")

            with result_col:
                st.subheader("Scenario Results")

                # Run scenario when button clicked
                if run_scenario:
                    if simulator is None:
                        st.error("Scenario simulator is not available. Please retrain the model and try again.")
                    else:
                        with st.spinner("Running scenario simulation..."):
                            # Create scenario config
                            from scenario_simulator import ScenarioConfig

                            scenario_config = ScenarioConfig(
                                name=scenario_name,
                                adjustments=adjustments,
                                description=f"Custom scenario with adjusted inputs"
                            )

                            # Run simulation
                            scenario_result = simulator.run_scenario(scenario_config)

                            # Store in session state
                            if 'scenarios' not in st.session_state:
                                st.session_state.scenarios = {}
                            st.session_state.scenarios[scenario_name] = scenario_result

                            st.success(f"Scenario '{scenario_name}' complete!")

                # Display results if scenarios exist
                if 'scenarios' in st.session_state and len(st.session_state.scenarios) > 0:
                    # Impact summary
                    st.markdown("#### Impact Summary")

                    latest_scenario = list(st.session_state.scenarios.values())[-1]
                    impact = latest_scenario.impact_summary

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Base Average",
                            f"${impact['base_average']:,.0f}"
                        )

                    with col2:
                        st.metric(
                            "Scenario Average",
                            f"${impact['scenario_average']:,.0f}",
                            delta=f"{impact['average_change_pct']:+.1f}%"
                        )

                    with col3:
                            st.metric(
                                "Base Total",
                                f"${impact['base_total']:,.0f}"
                            )
                        
                    with col4:
                        st.metric(
                            "Scenario Total",
                            f"${impact['scenario_total']:,.0f}",
                            delta=f"{impact['total_change_pct']:+.1f}%"
                        )

                    if latest_scenario.adjusted_drivers:
                        adjusted_driver_text = ", ".join(
                            f"{name}={value:,.2f}" for name, value in latest_scenario.adjusted_drivers.items()
                        )
                        st.caption(f"Scenario assumptions applied to the latest known driver values: {adjusted_driver_text}.")

                    st.info(
                        f"Compared with the base forecast, this scenario changes the average prediction by "
                        f"{impact['average_change_pct']:+.1f}% and the total forecast by {impact['total_change_pct']:+.1f}%. "
                        f"The largest upside shift is {impact['peak_change']:,.2f}, while the largest downside shift is "
                        f"{impact['trough_change']:,.2f}."
                    )

                    # Comparison chart
                    st.markdown("#### Base vs Scenario Comparison")

                    fig = go.Figure()

                    # Base case
                    fig.add_trace(go.Scatter(
                        x=result.future_predictions['date'],
                        y=result.future_predictions['predicted'],
                        mode='lines+markers',
                        name='Base Case',
                        line=dict(color='blue', width=2)
                    ))

                    # Scenario
                    fig.add_trace(go.Scatter(
                        x=latest_scenario.predictions['date'],
                        y=latest_scenario.predictions['predicted'],
                        mode='lines+markers',
                        name=latest_scenario.scenario_name,
                        line=dict(color='orange', width=2, dash='dash')
                    ))

                    fig.update_layout(
                        title=f"Forecast Comparison: Base v {latest_scenario.scenario_name}",
                        xaxis_title='Date',
                        yaxis_title=target_var,
                        hovermode='x unified',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Scenario comparison table
                    if len(st.session_state.scenarios) > 1:
                        st.markdown("#### All Scenarios Comparison")

                        comparison_df = simulator.compare_scenarios(
                            list(st.session_state.scenarios.keys())
                        )

                        # Format the dataframe
                        styled_df = comparison_df.copy()
                        styled_df['Average Prediction'] = styled_df['Average Prediction'].apply(lambda x: f"${x:,.0f}")
                        styled_df['Change from Base'] = styled_df['Change from Base'].apply(lambda x: f"${x:,.0f}")
                        styled_df['Change %'] = styled_df['Change %'].apply(lambda x: f"{x:+.2f}%")
                            
                        st.dataframe(styled_df, use_container_width=True)

                    # Clear scenarios button
                    if st.button("Clear All Scenarios"):
                        st.session_state.scenarios = {}
                        st.rerun()

                else:
                    st.info("Adjust the sliders and clici=k 'Run Scenario' to see results" ) 
            
            # Model summary
            st.markdown("---")
            st.subheader("📝 Model Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Information:**")
                st.write(f"- Training samples: {summary['training_samples']}")
                st.write(f"- Test samples: {summary['test_samples']}")
                st.write(f"- Forecast periods: {summary['forecast_periods']}")
                st.write(f"- Target variable: {summary['target_variable']}")
            
            with col2:
                st.markdown("**Model Performance:**")
                for metric, value in summary['model_performance'].items():
                    st.write(f"- {metric}: {value}")
            
            st.markdown("**Top 5 Features:**")
            st.write(", ".join(summary['top_features']))
