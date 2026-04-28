"""
Data Loader Module for Business Performance Forecaster
Handles CSV/Excel uploads, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial validation of business datasets"""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}
        
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or Excel file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        logger.info(f"Loading file: {file_path}")
        
        try:
            if path.suffix == '.csv':
                self.data = pd.read_csv(file_path)
            else:  # Excel files
                self.data = pd.read_excel(file_path)
            
            self._extract_metadata()
            logger.info(f"Successfully loaded {len(self.data)} rows, {len(self.data.columns)} columns")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def _extract_metadata(self):
        """Extract basic metadata from loaded data"""
        if self.data is None:
            return
        
        self.metadata = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns),
            'date_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the loaded data"""
        if self.data is None:
            return {}
        
        summary = {
            'basic_info': self.metadata,
            'numeric_stats': self.data.describe().to_dict() if len(self.metadata['numeric_columns']) > 0 else {},
            'missing_percentage': {
                col: (count / len(self.data)) * 100 
                for col, count in self.metadata['missing_values'].items()
            }
        }
        
        return summary
    
    def preview_data(self, n_rows: int = 5) -> pd.DataFrame:
        """Get first n rows of data for preview"""
        if self.data is None:
            return pd.DataFrame()
        return self.data.head(n_rows)


class DataValidator:
    """Validates business datasets for forecasting requirements"""
    
    def __init__(self, min_rows: int = 10):
        self.min_rows = min_rows
        self.validation_results: Dict = {}
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate if dataset meets minimum requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check minimum rows
        if len(df) < self.min_rows:
            issues.append(f"Dataset has only {len(df)} rows. Minimum {self.min_rows} required.")
        
        # Check for empty dataset
        if df.empty:
            issues.append("Dataset is empty.")
            return False, issues
        
        # Check for all null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            issues.append(f"Columns with all null values: {', '.join(all_null_cols)}")
        
        # Check for duplicate column names
        if df.columns.duplicated().any():
            issues.append("Dataset contains duplicate column names.")
        
        # Check for at least one numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            issues.append("Dataset must contain at least one numeric column for forecasting.")
        
        is_valid = len(issues) == 0
        
        self.validation_results = {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': self._generate_warnings(df)
        }
        
        return is_valid, issues
    
    def _generate_warnings(self, df: pd.DataFrame) -> List[str]:
        """Generate warnings for potential data quality issues"""
        warnings = []
        
        # High missing value percentage
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 30].index.tolist()
        if high_missing:
            warnings.append(
                f"Columns with >30% missing values: {', '.join(high_missing)}"
            )
        
        # Low variance numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if numeric_df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has constant value (no variance)")
        
        return warnings
    
    def check_time_series_requirements(self, df: pd.DataFrame, date_column: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Check if dataset meets time-series forecasting requirements
        
        Args:
            df: DataFrame to check
            date_column: Name of the date/time column (auto-detect if None)
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Try to find date column if not provided
        if date_column is None:
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Also check for columns that might be parseable as dates
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day']):
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                    except:
                        pass
            
            if len(date_cols) == 0:
                issues.append("No date/time column found. Time-series forecasting requires temporal data.")
                return False, issues
            
            date_column = date_cols[0]
            issues.append(f"INFO: Auto-detected date column: '{date_column}'")
        
        # Check if date column exists
        if date_column not in df.columns:
            issues.append(f"Specified date column '{date_column}' not found in dataset.")
            return False, issues
        
        # Try to parse dates
        try:
            dates = pd.to_datetime(df[date_column])
            
            # Check for sufficient time periods
            if len(dates.unique()) < 5:
                issues.append("Dataset has fewer than 5 unique time periods. More data needed for reliable forecasting.")
            
            # Check for duplicates
            if dates.duplicated().any():
                issues.append(f"Date column '{date_column}' contains duplicate timestamps.")
            
        except Exception as e:
            issues.append(f"Could not parse '{date_column}' as dates: {str(e)}")
            return False, issues
        
        is_valid = not any("No date/time column" in issue or "Could not parse" in issue for issue in issues)
        return is_valid, issues


class DataPreprocessor:
    """Preprocesses data for ML forecasting"""
    
    def __init__(self):
        self.transformations_applied: List[str] = []
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df: DataFrame to process
            strategy: 'auto', 'drop', 'forward_fill', 'mean', 'median'
            
        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()
        
        if strategy == 'auto':
            # For numeric columns, use forward fill then median
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_copy[col] = df_copy[col].ffill().fillna(df_copy[col].median())
            
            # For categorical, use forward fill then mode
            cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                df_copy[col] = df_copy[col].ffill().fillna(df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown')
            
            self.transformations_applied.append("Auto-filled missing values (numeric: forward-fill→median, categorical: forward-fill→mode)")
        
        elif strategy == 'drop':
            df_copy = df_copy.dropna()
            self.transformations_applied.append(f"Dropped rows with missing values ({len(df) - len(df_copy)} rows removed)")
        
        elif strategy == 'forward_fill':
            df_copy = df_copy.ffill().bfill()
            self.transformations_applied.append("Applied forward fill (then backward fill for remaining)")
        
        elif strategy == 'mean':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            self.transformations_applied.append("Filled numeric missing values with column means")
        
        elif strategy == 'median':
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
            self.transformations_applied.append("Filled numeric missing values with column medians")
        
        return df_copy
    
    def detect_and_convert_dates(self, df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect and convert date columns to datetime
        
        Args:
            df: DataFrame to process
            date_columns: List of column names to convert (auto-detect if None)
            
        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()
        converted_cols = []
        
        if date_columns is None:
            # Auto-detect potential date columns
            date_columns = []
            for col in df_copy.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day']):
                    date_columns.append(col)
        
        for col in date_columns:
            if col in df_copy.columns:
                try:
                    df_copy[col] = pd.to_datetime(df_copy[col])
                    converted_cols.append(col)
                except Exception as e:
                    logger.warning(f"Could not convert '{col}' to datetime: {str(e)}")
        
        if converted_cols:
            self.transformations_applied.append(f"Converted to datetime: {', '.join(converted_cols)}")
        
        return df_copy

    def auto_convert_numeric_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert text columns containing currency-formatted or comma-formatted numbers.

        Example: "$5,320" -> 5320
        """
        df_copy = df.copy()
        converted_cols = []

        for col in df_copy.select_dtypes(include=['object', 'string']).columns:
            original_series = df_copy[col]
            non_null_count = original_series.notna().sum()

            if non_null_count == 0:
                continue

            cleaned = original_series.astype(str).str.strip()
            cleaned = cleaned.str.replace(r'[\$,£€,]', '', regex=True)
            cleaned = cleaned.str.replace(r'\(([^)]+)\)', r'-\1', regex=True)

            converted = pd.to_numeric(cleaned, errors='coerce')
            success_ratio = converted.notna().sum() / non_null_count

            if success_ratio >= 0.8:
                df_copy[col] = converted
                converted_cols.append(col)

        if converted_cols:
            self.transformations_applied.append(
                f"Converted numeric-like text columns: {', '.join(converted_cols)}"
            )

        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numeric columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to check (all numeric if None)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
            
        Returns:
            Processed DataFrame
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(df_copy)
        
        if method == 'iqr':
            for col in columns:
                if col in df_copy.columns:
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                if col in df_copy.columns:
                    z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                    df_copy = df_copy[z_scores < threshold]
        
        rows_removed = initial_rows - len(df_copy)
        if rows_removed > 0:
            self.transformations_applied.append(f"Removed {rows_removed} outlier rows using {method.upper()} method")
        
        return df_copy
    
    def get_transformation_summary(self) -> List[str]:
        """Get list of all transformations applied"""
        return self.transformations_applied.copy()
