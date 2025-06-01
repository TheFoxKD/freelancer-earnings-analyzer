"""
Main CLI interface for Freelancer Analyzer.

This module provides the command-line interface for the freelancer
earnings analysis system with natural language processing capabilities.
"""

import click
import os
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from dotenv import load_dotenv

from .data_loader import DataLoader
from .data_analyzer import DataAnalyzer
from .llm_integration import SimpleLLMProcessor
from .utils import format_data_as_json

# Load environment variables from .env file
load_dotenv()

console = Console()


class FreelancerAnalyzer:
    """
    Main application class for freelancer data analysis.

    Coordinates data loading, analysis, and LLM integration.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the analyzer system.

        Args:
            data_path: Path to the freelancer data CSV file (if None, uses DATA_PATH from .env or default)
        """
        # Use data_path from parameter, environment variable, or default
        self.data_path: str = (
            data_path
            if data_path is not None
            else os.getenv("DATA_PATH", "data/freelancer_earnings_bd.csv")
        )
        self.data_loader: Optional[DataLoader] = None
        self.data_analyzer: Optional[DataAnalyzer] = None
        self.llm_processor: Optional[SimpleLLMProcessor] = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize all system components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            console.print(
                "[bold blue]🚀 Initializing Freelancer Analyzer...[/bold blue]"
            )

            # Initialize data loader
            self.data_loader = DataLoader(self.data_path)
            self.data_loader.load_data()

            # Initialize data analyzer
            self.data_analyzer = DataAnalyzer(self.data_loader)

            # Initialize LLM processor
            self.llm_processor = SimpleLLMProcessor(self.data_analyzer)

            self.initialized = True
            console.print(
                "[bold green]✅ System initialized successfully![/bold green]"
            )
            return True

        except Exception as e:
            console.print(f"[bold red]❌ Initialization failed: {e}[/bold red]")
            return False

    def display_welcome_message(self):
        """Display welcome message and available commands."""
        welcome_text = """
🎯 Freelancer Earnings Analyzer

Система анализа данных о доходах фрилансеров с поддержкой запросов на естественном языке.

Доступные команды:
• ask - Задать вопрос на естественном языке
• analyze - Запустить конкретный тип анализа
• samples - Показать примеры вопросов
• health - Проверить статус системы
• info - Показать информацию о данных
• exit - Выйти из программы
        """

        console.print(
            Panel(welcome_text, title="Добро пожаловать", border_style="blue")
        )

    def ask_question(self, question: Optional[str] = None):
        """
        Process a natural language question.

        Args:
            question: User question (if None, will prompt for input)
        """
        if not question:
            question = Prompt.ask("\n[bold blue]Задайте ваш вопрос[/bold blue]")

        if not question.strip():
            console.print("[yellow]⚠️  Пожалуйста, введите вопрос[/yellow]")
            return

        console.print(f"\n[bold]🤔 Обрабатываю вопрос:[/bold] {question}")

        # Process the question
        if not self.llm_processor:
            console.print("[red]❌ LLM processor not initialized[/red]")
            return

        result = self.llm_processor.process_question(question)

        if result["status"] == "success":
            # Display LLM response
            console.print("\n[bold green]📋 Ответ:[/bold green]")
            console.print(Panel(result["llm_response"], border_style="green"))

            # Optionally show raw data
            show_data = Prompt.ask(
                "\nПоказать детальные данные анализа? (y/N)", default="n"
            )
            if show_data.lower() in ["y", "yes", "да"]:
                self._display_analysis_data(result["analysis_data"])
        else:
            console.print(
                f"[bold red]❌ Ошибка:[/bold red] {result.get('error', 'Unknown error')}"
            )
            if "fallback_response" in result:
                console.print(result["fallback_response"])

    def run_specific_analysis(self, analysis_type: Optional[str] = None):
        """
        Run a specific type of analysis.

        Args:
            analysis_type: Type of analysis to run
        """
        available_analyses = {
            "1": ("crypto_payment", "Анализ доходов от криптоплатежей"),
            "2": ("regional_income", "Распределение доходов по регионам"),
            "3": ("expert_projects", "Анализ проектов экспертов"),
            "4": ("experience_rates", "Связь опыта и ставок"),
            "5": ("specialization_earnings", "Доходы по специализациям"),
            "6": ("platform_performance", "Производительность платформ"),
            "7": ("summary", "Общая сводка"),
        }

        if not analysis_type:
            console.print("\n[bold]📊 Выберите тип анализа:[/bold]")
            for key, (_, description) in available_analyses.items():
                console.print(f"  {key}. {description}")

            choice = Prompt.ask(
                "\nВведите номер", choices=list(available_analyses.keys())
            )
            analysis_type = available_analyses[choice][0]

        # Run the analysis
        console.print(f"\n[bold blue]🔍 Запускаю анализ: {analysis_type}[/bold blue]")

        try:
            analysis_function = getattr(
                self.data_analyzer,
                f"analyze_{analysis_type}"
                if analysis_type != "summary"
                else "get_comprehensive_summary",
            )
            result = analysis_function()
            self._display_analysis_data(result)

        except Exception as e:
            console.print(f"[bold red]❌ Ошибка анализа: {e}[/bold red]")

    def show_sample_questions(self):
        """Display sample questions that can be asked."""
        samples = self.llm_processor.get_sample_questions()

        console.print("\n[bold blue]📝 Примеры вопросов:[/bold blue]")
        for i, question in enumerate(samples, 1):
            console.print(f"  {i}. {question}")

        # Allow user to try a sample question
        try_sample = Prompt.ask(
            "\nХотите попробовать один из примеров? Введите номер (или Enter для отмены)",
            default="",
        )
        if try_sample.isdigit() and 1 <= int(try_sample) <= len(samples):
            selected_question = samples[int(try_sample) - 1]
            console.print(f"\n[bold]Выбранный вопрос:[/bold] {selected_question}")
            self.ask_question(selected_question)

    def show_health_status(self):
        """Display system health status."""
        health = self.llm_processor.health_check()

        console.print("\n[bold blue]🏥 Статус системы:[/bold blue]")

        # Overall status
        status_color = (
            "green"
            if health["overall_status"] == "healthy"
            else "yellow"
            if health["overall_status"] == "degraded"
            else "red"
        )
        console.print(
            f"Общий статус: [{status_color}]{health['overall_status']}[/{status_color}]"
        )

        # Data analyzer status
        console.print(
            f"Анализатор данных: [green]{'✅' if health['data_analyzer'] else '❌'}[/green]"
        )

        # LLM service status
        console.print("\nСтатус LLM сервиса:")
        anthropic_icon = "✅" if health.get("anthropic_available", False) else "❌"
        console.print(f"  Anthropic SDK: {anthropic_icon}")

        api_key_icon = "✅" if health.get("anthropic_api_key_set", False) else "❌"
        console.print(f"  Anthropic API Key: {api_key_icon}")

        llm_init_icon = "✅" if health.get("llm_initialized", False) else "❌"
        console.print(f"  LLM инициализирован: {llm_init_icon}")

        if "llm_test" in health:
            test_status = health["llm_test"]
            test_icon = "✅" if test_status == "passed" else "❌"
            console.print(f"  LLM тест: {test_icon} {test_status}")

        if not health.get("llm_initialized", False):
            console.print("  [yellow]⚠️  LLM модель недоступна[/yellow]")

    def show_data_info(self):
        """Display information about the loaded dataset."""
        info = self.data_loader.get_data_info()
        self.data_loader.get_basic_stats()
        quality = self.data_loader.validate_data_quality()

        # Dataset overview
        console.print("\n[bold blue]📈 Информация о датасете:[/bold blue]")
        overview_table = Table(title="Обзор данных")
        overview_table.add_column("Параметр", style="cyan")
        overview_table.add_column("Значение", style="magenta")

        overview_table.add_row("Общее количество записей", str(info["total_records"]))
        overview_table.add_row("Количество колонок", str(len(info["columns"])))
        overview_table.add_row(
            "Платформы", str(len(info["categorical_columns"]["Platform"]))
        )
        overview_table.add_row(
            "Категории работ", str(len(info["categorical_columns"]["Job_Category"]))
        )
        overview_table.add_row(
            "Регионы", str(len(info["categorical_columns"]["Client_Region"]))
        )

        console.print(overview_table)

        # Data quality
        console.print("\n[bold green]🔍 Качество данных:[/bold green]")
        quality_table = Table(title="Проверка качества")
        quality_table.add_column("Метрика", style="cyan")
        quality_table.add_column("Значение", style="magenta")

        quality_table.add_row("Дубликаты ID", str(quality["duplicate_freelancer_ids"]))
        quality_table.add_row(
            "Записи с пропусками", str(quality["records_with_missing_values"])
        )
        quality_table.add_row(
            "Нулевые доходы", str(quality["earnings_anomalies"]["zero_earnings"])
        )
        quality_table.add_row(
            "Отрицательные доходы",
            str(quality["earnings_anomalies"]["negative_earnings"]),
        )

        console.print(quality_table)

    def _display_analysis_data(self, data: dict):
        """
        Display analysis data in a formatted way.

        Args:
            data: Analysis results dictionary
        """
        console.print("\n[bold green]📊 Детальные результаты анализа:[/bold green]")

        # Convert to pretty JSON and display
        json_str = format_data_as_json(data, indent=2)
        console.print(Panel(json_str, title="Данные анализа", border_style="green"))


@click.group()
def cli():
    """Freelancer Earnings Analyzer - AI-powered data analysis system."""
    pass


@cli.command()
@click.option(
    "--data-path",
    "-d",
    default=None,
    help="Path to the data file (default: from .env or data/freelancer_earnings_bd.csv)",
)
def interactive(data_path):
    """Start interactive mode for asking questions and running analysis."""
    analyzer = FreelancerAnalyzer(data_path)

    if not analyzer.initialize():
        return

    analyzer.display_welcome_message()

    while True:
        try:
            command = Prompt.ask(
                "\n[bold cyan]Введите команду[/bold cyan]",
                choices=["ask", "analyze", "samples", "health", "info", "exit"],
                default="ask",
            )

            if command == "ask":
                analyzer.ask_question()
            elif command == "analyze":
                analyzer.run_specific_analysis()
            elif command == "samples":
                analyzer.show_sample_questions()
            elif command == "health":
                analyzer.show_health_status()
            elif command == "info":
                analyzer.show_data_info()
            elif command == "exit":
                console.print(
                    "\n[bold blue]👋 Спасибо за использование Freelancer Analyzer![/bold blue]"
                )
                break

        except KeyboardInterrupt:
            console.print("\n\n[bold blue]👋 Выход из программы...[/bold blue]")
            break
        except Exception as e:
            console.print(f"\n[bold red]❌ Ошибка: {e}[/bold red]")


@cli.command()
@click.argument("question")
@click.option(
    "--data-path",
    "-d",
    default=None,
    help="Path to the data file (default: from .env or data/freelancer_earnings_bd.csv)",
)
def ask(question, data_path):
    """Ask a single question and get an answer."""
    analyzer = FreelancerAnalyzer(data_path)

    if not analyzer.initialize():
        return

    analyzer.ask_question(question)


@cli.command()
@click.option(
    "--data-path",
    "-d",
    default=None,
    help="Path to the data file (default: from .env or data/freelancer_earnings_bd.csv)",
)
def info(data_path):
    """Show information about the dataset."""
    analyzer = FreelancerAnalyzer(data_path)

    if not analyzer.initialize():
        return

    analyzer.show_data_info()


@cli.command()
@click.option(
    "--data-path",
    "-d",
    default=None,
    help="Path to the data file (default: from .env or data/freelancer_earnings_bd.csv)",
)
def health(data_path):
    """Check system health status."""
    analyzer = FreelancerAnalyzer(data_path)

    if not analyzer.initialize():
        return

    analyzer.show_health_status()


if __name__ == "__main__":
    cli()
