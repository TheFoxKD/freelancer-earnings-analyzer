"""
Data analysis module for freelancer earnings analysis.

This module provides specialized analysis functions to answer specific
questions about freelancer earnings and performance metrics.
"""

import numpy as np
from typing import Dict, Any
from .data_loader import DataLoader


class DataAnalyzer:
    """
    Provides analytical functions for freelancer earnings data.

    This class contains methods to analyze various aspects of freelancer
    earnings, performance, and market trends.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize the analyzer with loaded data.

        Args:
            data_loader: DataLoader instance with loaded data
        """
        self.data_loader = data_loader
        if not data_loader._data_loaded or data_loader.df is None:
            raise ValueError("DataLoader must have data loaded before analysis")
        self.df = data_loader.df

    def analyze_crypto_payment_earnings(self) -> Dict[str, Any]:
        """
        Analyze earnings difference between crypto and other payment methods.

        Answers: "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте,
        по сравнению с другими способами оплаты?"

        Returns:
            Dictionary with crypto vs non-crypto earnings analysis
        """
        # Separate crypto and non-crypto payments
        crypto_earnings = self.df[self.df["Payment_Method"] == "Crypto"]["Earnings_USD"]
        other_earnings = self.df[self.df["Payment_Method"] != "Crypto"]["Earnings_USD"]

        # Calculate statistics
        crypto_stats = {
            "mean": round(crypto_earnings.mean(), 2),
            "median": round(crypto_earnings.median(), 2),
            "count": len(crypto_earnings),
            "std": round(crypto_earnings.std(), 2),
        }

        other_stats = {
            "mean": round(other_earnings.mean(), 2),
            "median": round(other_earnings.median(), 2),
            "count": len(other_earnings),
            "std": round(other_earnings.std(), 2),
        }

        # Calculate percentage difference
        mean_diff_percent = round(
            ((crypto_stats["mean"] - other_stats["mean"]) / other_stats["mean"]) * 100,
            2,
        )
        median_diff_percent = round(
            ((crypto_stats["median"] - other_stats["median"]) / other_stats["median"])
            * 100,
            2,
        )

        # Breakdown by payment method
        payment_breakdown = (
            self.df.groupby("Payment_Method")["Earnings_USD"]
            .agg(["mean", "median", "count", "std"])
            .round(2)
        )

        return {
            "crypto_vs_others": {
                "crypto_earnings": crypto_stats,
                "other_earnings": other_stats,
                "mean_difference_percent": mean_diff_percent,
                "median_difference_percent": median_diff_percent,
                "crypto_higher": crypto_stats["mean"] > other_stats["mean"],
            },
            "payment_method_breakdown": payment_breakdown.to_dict("index"),
            "summary": f"Crypto payments {'higher' if mean_diff_percent > 0 else 'lower'} by {abs(mean_diff_percent):.1f}% on average",
        }

    def analyze_regional_income_distribution(self) -> Dict[str, Any]:
        """
        Analyze income distribution by client region.

        Answers: "Как распределяется доход фрилансеров в зависимости от региона проживания?"

        Returns:
            Dictionary with regional income analysis
        """
        # Regional statistics
        regional_stats = (
            self.df.groupby("Client_Region")["Earnings_USD"]
            .agg(["mean", "median", "count", "std", "min", "max"])
            .round(2)
        )

        # Calculate percentiles for each region
        regional_percentiles = {}
        for region in self.df["Client_Region"].unique():
            region_data = self.df[self.df["Client_Region"] == region]["Earnings_USD"]
            regional_percentiles[region] = {
                "p25": round(np.percentile(region_data, 25), 2),
                "p50": round(np.percentile(region_data, 50), 2),
                "p75": round(np.percentile(region_data, 75), 2),
                "p90": round(np.percentile(region_data, 90), 2),
            }

        # Find highest and lowest earning regions
        mean_earnings = regional_stats["mean"].to_dict()
        highest_region = max(mean_earnings, key=mean_earnings.get)  # type: ignore[arg-type]
        lowest_region = min(mean_earnings, key=mean_earnings.get)  # type: ignore[arg-type]

        # Calculate regional market share
        regional_counts = self.df["Client_Region"].value_counts()
        regional_market_share = (
            (regional_counts / len(self.df) * 100).round(2).to_dict()
        )

        return {
            "regional_statistics": regional_stats.to_dict("index"),
            "regional_percentiles": regional_percentiles,
            "market_leaders": {
                "highest_earning_region": {
                    "region": highest_region,
                    "average_earnings": mean_earnings[highest_region],
                },
                "lowest_earning_region": {
                    "region": lowest_region,
                    "average_earnings": mean_earnings[lowest_region],
                },
            },
            "market_share_by_region": regional_market_share,
            "earnings_range": {
                "global_min": round(self.df["Earnings_USD"].min(), 2),
                "global_max": round(self.df["Earnings_USD"].max(), 2),
                "global_mean": round(self.df["Earnings_USD"].mean(), 2),
            },
        }

    def analyze_expert_projects_completion(self) -> Dict[str, Any]:
        """
        Analyze project completion patterns for expert-level freelancers.

        Answers: "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?"

        Returns:
            Dictionary with expert freelancers analysis
        """
        # Filter expert freelancers
        experts = self.df[self.df["Experience_Level"] == "Expert"]

        # Count experts with less than 100 projects
        experts_under_100 = experts[experts["Job_Completed"] < 100]
        experts_100_plus = experts[experts["Job_Completed"] >= 100]

        # Calculate percentages
        total_experts = len(experts)
        under_100_count = len(experts_under_100)
        under_100_percentage = (
            round((under_100_count / total_experts) * 100, 2)
            if total_experts > 0
            else 0
        )

        # Breakdown by project ranges
        project_ranges = {
            "0-49": len(experts[experts["Job_Completed"] < 50]),
            "50-99": len(
                experts[
                    (experts["Job_Completed"] >= 50) & (experts["Job_Completed"] < 100)
                ]
            ),
            "100-199": len(
                experts[
                    (experts["Job_Completed"] >= 100) & (experts["Job_Completed"] < 200)
                ]
            ),
            "200+": len(experts[experts["Job_Completed"] >= 200]),
        }

        # Compare with other experience levels
        experience_project_stats = (
            self.df.groupby("Experience_Level")["Job_Completed"]
            .agg(["mean", "median", "count", "min", "max"])
            .round(2)
        )

        # Analyze expert performance metrics
        expert_performance = {
            "average_earnings": round(experts["Earnings_USD"].mean(), 2),
            "average_hourly_rate": round(experts["Hourly_Rate"].mean(), 2),
            "average_success_rate": round(experts["Job_Success_Rate"].mean(), 2),
            "average_client_rating": round(experts["Client_Rating"].mean(), 2),
        }

        return {
            "expert_project_analysis": {
                "total_experts": total_experts,
                "experts_under_100_projects": under_100_count,
                "percentage_under_100": under_100_percentage,
                "experts_100_plus_projects": len(experts_100_plus),
            },
            "project_range_breakdown": project_ranges,
            "experience_level_comparison": experience_project_stats.to_dict("index"),
            "expert_performance_metrics": expert_performance,
            "insights": {
                "expert_completion_rate": f"{under_100_percentage}% of experts have completed less than 100 projects",
                "most_productive_experts": len(
                    experts[experts["Job_Completed"] >= 200]
                ),
            },
        }

    def analyze_experience_vs_rates(self) -> Dict[str, Any]:
        """
        Analyze relationship between experience level and hourly rates.

        Additional analysis: Связь между опытом работы и ставками

        Returns:
            Dictionary with experience vs rates analysis
        """
        # Calculate simple statistics by experience level
        exp_levels = ["Beginner", "Intermediate", "Expert"]
        experience_stats = {}
        rate_progression = {}

        for level in exp_levels:
            if level in self.df["Experience_Level"].unique():
                level_data = self.df[self.df["Experience_Level"] == level]

                # Calculate detailed statistics for this level
                experience_stats[level] = {
                    "hourly_rate_mean": round(level_data["Hourly_Rate"].mean(), 2),
                    "hourly_rate_median": round(level_data["Hourly_Rate"].median(), 2),
                    "hourly_rate_std": round(level_data["Hourly_Rate"].std(), 2),
                    "hourly_rate_min": round(level_data["Hourly_Rate"].min(), 2),
                    "hourly_rate_max": round(level_data["Hourly_Rate"].max(), 2),
                    "earnings_mean": round(level_data["Earnings_USD"].mean(), 2),
                    "earnings_median": round(level_data["Earnings_USD"].median(), 2),
                    "success_rate_mean": round(
                        level_data["Job_Success_Rate"].mean(), 2
                    ),
                    "client_rating_mean": round(level_data["Client_Rating"].mean(), 2),
                    "freelancer_count": len(level_data),
                }

                # Simplified progression data
                rate_progression[level] = {
                    "avg_hourly_rate": round(level_data["Hourly_Rate"].mean(), 2),
                    "avg_earnings": round(level_data["Earnings_USD"].mean(), 2),
                    "freelancer_count": len(level_data),
                }

        return {
            "experience_statistics": experience_stats,
            "rate_progression": rate_progression,
            "skill_premium": {
                "expert_vs_beginner_rate": round(
                    (
                        rate_progression.get("Expert", {}).get("avg_hourly_rate", 0)
                        - rate_progression.get("Beginner", {}).get("avg_hourly_rate", 0)
                    ),
                    2,
                ),
                "expert_vs_intermediate_rate": round(
                    (
                        rate_progression.get("Expert", {}).get("avg_hourly_rate", 0)
                        - rate_progression.get("Intermediate", {}).get(
                            "avg_hourly_rate", 0
                        )
                    ),
                    2,
                ),
            },
        }

    def analyze_specialization_earnings(self) -> Dict[str, Any]:
        """
        Analyze earnings by job category/specialization.

        Additional analysis: Влияние специализации на доходы

        Returns:
            Dictionary with specialization earnings analysis
        """
        # Calculate statistics for each category
        categories = self.df["Job_Category"].unique()
        category_stats = {}

        for category in categories:
            cat_data = self.df[self.df["Job_Category"] == category]
            category_stats[category] = {
                "earnings_mean": round(cat_data["Earnings_USD"].mean(), 2),
                "earnings_median": round(cat_data["Earnings_USD"].median(), 2),
                "earnings_std": round(cat_data["Earnings_USD"].std(), 2),
                "hourly_rate_mean": round(cat_data["Hourly_Rate"].mean(), 2),
                "hourly_rate_median": round(cat_data["Hourly_Rate"].median(), 2),
                "success_rate_mean": round(cat_data["Job_Success_Rate"].mean(), 2),
                "client_rating_mean": round(cat_data["Client_Rating"].mean(), 2),
                "freelancer_count": len(cat_data),
            }

        # Find highest and lowest paying categories
        mean_earnings_by_category = self.df.groupby("Job_Category")[
            "Earnings_USD"
        ].mean()
        highest_paying = mean_earnings_by_category.idxmax()
        lowest_paying = mean_earnings_by_category.idxmin()

        # Market demand analysis
        category_demand = self.df["Job_Category"].value_counts()
        category_market_share = (category_demand / len(self.df) * 100).round(2)

        return {
            "category_statistics": category_stats,
            "market_leaders": {
                "highest_paying_category": {
                    "category": highest_paying,
                    "average_earnings": round(
                        mean_earnings_by_category[highest_paying], 2
                    ),
                },
                "lowest_paying_category": {
                    "category": lowest_paying,
                    "average_earnings": round(
                        mean_earnings_by_category[lowest_paying], 2
                    ),
                },
            },
            "market_demand": category_demand.to_dict(),
            "market_share_percentage": category_market_share.to_dict(),
        }

    def analyze_platform_performance(self) -> Dict[str, Any]:
        """
        Analyze performance metrics across different freelancing platforms.

        Additional analysis: Platform comparison

        Returns:
            Dictionary with platform performance analysis
        """
        # Calculate statistics for each platform
        platforms = self.df["Platform"].unique()
        platform_stats = {}

        for platform in platforms:
            platform_data = self.df[self.df["Platform"] == platform]
            platform_stats[platform] = {
                "earnings_mean": round(platform_data["Earnings_USD"].mean(), 2),
                "earnings_median": round(platform_data["Earnings_USD"].median(), 2),
                "hourly_rate_mean": round(platform_data["Hourly_Rate"].mean(), 2),
                "hourly_rate_median": round(platform_data["Hourly_Rate"].median(), 2),
                "success_rate_mean": round(platform_data["Job_Success_Rate"].mean(), 2),
                "client_rating_mean": round(platform_data["Client_Rating"].mean(), 2),
                "rehire_rate_mean": round(platform_data["Rehire_Rate"].mean(), 2),
                "freelancer_count": len(platform_data),
            }

        # Platform market share
        platform_counts = self.df["Platform"].value_counts()
        platform_market_share = (platform_counts / len(self.df) * 100).round(2)

        return {
            "platform_statistics": platform_stats,
            "market_share": platform_market_share.to_dict(),
            "platform_ranking": {
                "by_earnings": self.df.groupby("Platform")["Earnings_USD"]
                .mean()
                .sort_values(ascending=False)
                .to_dict(),
                "by_success_rate": self.df.groupby("Platform")["Job_Success_Rate"]
                .mean()
                .sort_values(ascending=False)
                .to_dict(),
            },
        }

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all key findings.

        Returns:
            Dictionary with overall summary statistics
        """
        return {
            "dataset_overview": {
                "total_freelancers": len(self.df),
                "average_earnings": round(self.df["Earnings_USD"].mean(), 2),
                "median_earnings": round(self.df["Earnings_USD"].median(), 2),
                "average_hourly_rate": round(self.df["Hourly_Rate"].mean(), 2),
                "average_success_rate": round(self.df["Job_Success_Rate"].mean(), 2),
            },
            "market_distribution": {
                "platforms": len(self.df["Platform"].unique()),
                "job_categories": len(self.df["Job_Category"].unique()),
                "regions": len(self.df["Client_Region"].unique()),
                "payment_methods": len(self.df["Payment_Method"].unique()),
            },
            "top_performers": {
                "highest_earner": round(self.df["Earnings_USD"].max(), 2),
                "highest_hourly_rate": round(self.df["Hourly_Rate"].max(), 2),
                "most_projects_completed": int(self.df["Job_Completed"].max()),
            },
        }
