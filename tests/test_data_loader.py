"""
Tests for data_loader module.

This module contains unit tests for the DataLoader class
to ensure proper data loading and validation.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import csv

from src.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init_with_default_path(self):
        """Test DataLoader initialization with default path."""
        loader = DataLoader()
        assert loader.data_path == Path("data/freelancer_earnings_bd.csv")
        assert loader.df is None
        assert loader._data_loaded is False

    def test_init_with_custom_path(self):
        """Test DataLoader initialization with custom path."""
        custom_path = "custom/path/data.csv"
        loader = DataLoader(custom_path)
        assert loader.data_path == Path(custom_path)

    def test_load_data_file_not_found(self):
        """Test load_data raises FileNotFoundError for non-existent file."""
        loader = DataLoader("non_existent_file.csv")

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_load_data_empty_file(self):
        """Test load_data raises EmptyDataError for empty CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            loader = DataLoader(temp_path)
            with pytest.raises(pd.errors.EmptyDataError):
                loader.load_data()
        finally:
            Path(temp_path).unlink()

    def test_load_data_missing_columns(self):
        """Test load_data raises ValueError for missing required columns."""
        # Create a CSV with some columns but missing required ones
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Freelancer_ID", "Job_Category"]
            )  # Missing required columns
            writer.writerow([1, "Web Development"])
            temp_path = f.name

        try:
            loader = DataLoader(temp_path)
            with pytest.raises(ValueError) as exc_info:
                loader.load_data()
            assert "Missing required columns" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_load_data_success(self):
        """Test successful data loading with valid CSV."""
        # Create a valid CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            # Write header with all required columns
            writer.writerow(
                [
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
                    "Job_Duration_Days",
                    "Project_Type",
                    "Rehire_Rate",
                    "Marketing_Spend",
                ]
            )
            # Write a sample data row
            writer.writerow(
                [
                    1,
                    "Web Development",
                    "Fiverr",
                    "Beginner",
                    "USA",
                    "PayPal",
                    50,
                    5000,
                    100,
                    85.5,
                    4.2,
                    30,
                    "Fixed",
                    60.0,
                    200,
                ]
            )
            temp_path = f.name

        try:
            loader = DataLoader(temp_path)
            df = loader.load_data()

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert loader._data_loaded is True
            assert loader.df is not None
        finally:
            Path(temp_path).unlink()

    def test_get_data_info_without_loading(self):
        """Test get_data_info raises error when data not loaded."""
        loader = DataLoader()

        with pytest.raises(ValueError) as exc_info:
            loader.get_data_info()
        assert "Data not loaded" in str(exc_info.value)

    def test_get_basic_stats_without_loading(self):
        """Test get_basic_stats raises error when data not loaded."""
        loader = DataLoader()

        with pytest.raises(ValueError) as exc_info:
            loader.get_basic_stats()
        assert "Data not loaded" in str(exc_info.value)

    def test_validate_data_quality_without_loading(self):
        """Test validate_data_quality raises error when data not loaded."""
        loader = DataLoader()

        with pytest.raises(ValueError) as exc_info:
            loader.validate_data_quality()
        assert "Data not loaded" in str(exc_info.value)


@pytest.fixture
def sample_data_loader():
    """Fixture that provides a DataLoader with sample data."""
    # Create sample data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
                "Job_Duration_Days",
                "Project_Type",
                "Rehire_Rate",
                "Marketing_Spend",
            ]
        )

        # Add sample rows
        sample_data = [
            [
                1,
                "Web Development",
                "Fiverr",
                "Expert",
                "USA",
                "Crypto",
                150,
                8000,
                120,
                90.0,
                4.5,
                25,
                "Fixed",
                70.0,
                300,
            ],
            [
                2,
                "Data Entry",
                "Upwork",
                "Beginner",
                "Asia",
                "PayPal",
                30,
                1500,
                50,
                75.0,
                3.8,
                15,
                "Hourly",
                40.0,
                100,
            ],
            [
                3,
                "Graphic Design",
                "Freelancer",
                "Intermediate",
                "Europe",
                "Bank Transfer",
                80,
                4500,
                90,
                85.0,
                4.2,
                20,
                "Fixed",
                65.0,
                250,
            ],
        ]

        for row in sample_data:
            writer.writerow(row)

        temp_path = f.name

    # Create and load data
    loader = DataLoader(temp_path)
    loader.load_data()

    yield loader

    # Cleanup
    Path(temp_path).unlink()


def test_get_data_info_with_data(sample_data_loader):
    """Test get_data_info with loaded data."""
    info = sample_data_loader.get_data_info()

    assert info["total_records"] == 3
    assert "Freelancer_ID" in info["columns"]
    assert (
        len(info["categorical_columns"]["Platform"]) == 3
    )  # Fiverr, Upwork, Freelancer
    assert (
        len(info["categorical_columns"]["Experience_Level"]) == 3
    )  # Expert, Beginner, Intermediate


def test_get_basic_stats_with_data(sample_data_loader):
    """Test get_basic_stats with loaded data."""
    stats = sample_data_loader.get_basic_stats()

    assert "Earnings_USD" in stats
    assert "Job_Completed" in stats
    assert stats["Earnings_USD"]["count"] == 3
    assert stats["Earnings_USD"]["mean"] > 0


def test_validate_data_quality_with_data(sample_data_loader):
    """Test validate_data_quality with loaded data."""
    quality = sample_data_loader.validate_data_quality()

    assert quality["total_records"] == 3
    assert quality["duplicate_freelancer_ids"] == 0
    assert quality["earnings_anomalies"]["negative_earnings"] == 0
