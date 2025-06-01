"""
Data loading and preprocessing module for freelancer earnings analysis.

This module handles loading the CSV data and provides basic data structure
information and validation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


class DataLoader:
    """
    Handles loading and basic preprocessing of freelancer earnings data.

    This class provides methods to load the CSV file, validate data structure,
    and get basic information about the dataset.
    """

    def __init__(self, data_path: str = "data/freelancer_earnings_bd.csv"):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the CSV file with freelancer data
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self._data_loaded = False

    def load_data(self) -> pd.DataFrame:
        """
        Load the freelancer earnings data from CSV file.

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        try:
            # Load the CSV file
            self.df = pd.read_csv(self.data_path)
            self._data_loaded = True

            # Basic data validation
            if self.df.empty:
                raise pd.errors.EmptyDataError("The CSV file is empty")

            # Check for required columns
            required_columns = [
                "Freelancer_ID",
                "Job_Category",
                "Platform",
                "Experience_Level",
                "Client_Region",
                "Payment_Method",
                "Job_Completed",
                "Earnings_USD",
                "Hourly_Rate",
                "Job_Success_Rate",
                "Client_Rating",
            ]

            missing_columns = [
                col for col in required_columns if col not in self.df.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            print(f"✅ Data loaded successfully: {len(self.df)} records")
            return self.df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded dataset.

        Returns:
            Dictionary with dataset information
        """
        if not self._data_loaded or self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        info = {
            "total_records": len(self.df),
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "unique_values": {col: self.df[col].nunique() for col in self.df.columns},
            "categorical_columns": {
                "Job_Category": sorted(self.df["Job_Category"].unique()),
                "Platform": sorted(self.df["Platform"].unique()),
                "Experience_Level": sorted(self.df["Experience_Level"].unique()),
                "Client_Region": sorted(self.df["Client_Region"].unique()),
                "Payment_Method": sorted(self.df["Payment_Method"].unique()),
                "Project_Type": sorted(self.df["Project_Type"].unique()),
            },
        }

        return info

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistical information about numerical columns.

        Returns:
            Dictionary with statistical information
        """
        if not self._data_loaded or self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        numerical_columns = [
            "Job_Completed",
            "Earnings_USD",
            "Hourly_Rate",
            "Job_Success_Rate",
            "Client_Rating",
            "Job_Duration_Days",
            "Rehire_Rate",
            "Marketing_Spend",
        ]

        stats = {}
        for col in numerical_columns:
            if col in self.df.columns:
                stats[col] = {
                    "mean": round(self.df[col].mean(), 2),
                    "median": round(self.df[col].median(), 2),
                    "std": round(self.df[col].std(), 2),
                    "min": round(self.df[col].min(), 2),
                    "max": round(self.df[col].max(), 2),
                    "count": self.df[col].count(),
                }

        return stats

    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks on the loaded dataset.

        Returns:
            Dictionary with data quality assessment
        """
        if not self._data_loaded or self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        quality_report = {
            "total_records": len(self.df),
            "duplicate_freelancer_ids": self.df["Freelancer_ID"].duplicated().sum(),
            "records_with_missing_values": self.df.isnull().any(axis=1).sum(),
            "earnings_anomalies": {
                "zero_earnings": (self.df["Earnings_USD"] == 0).sum(),
                "negative_earnings": (self.df["Earnings_USD"] < 0).sum(),
                "extremely_high_earnings": (self.df["Earnings_USD"] > 10000).sum(),
            },
            "rating_anomalies": {
                "out_of_range_ratings": (
                    (self.df["Client_Rating"] < 1) | (self.df["Client_Rating"] > 5)
                ).sum()
            },
        }

        return quality_report
