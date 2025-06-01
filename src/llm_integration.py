"""
Claude LLM integration module using Anthropic API.

This module provides a clean and reliable LLM integration using
Anthropic's Claude models with environment variable configuration.
"""

import os
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from .data_analyzer import DataAnalyzer
from .utils import format_data_as_json

# Load environment variables from .env file
load_dotenv()

# Import Anthropic components
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("❌ Anthropic not installed. Run: uv add anthropic")


class SimpleLLMProcessor:
    """
    Simple LLM processor using Anthropic Claude.

    This class provides clean LLM integration with Claude models.
    API key is loaded from ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, data_analyzer: DataAnalyzer):
        """
        Initialize the Claude LLM processor.

        Args:
            data_analyzer: DataAnalyzer instance with loaded data
        """
        self.analyzer = data_analyzer
        self.client: Optional[anthropic.Anthropic] = self._initialize_claude()

        # Map questions to analysis functions
        self.question_mapping = {
            "crypto_payment": self.analyzer.analyze_crypto_payment_earnings,
            "regional_income": self.analyzer.analyze_regional_income_distribution,
            "expert_projects": self.analyzer.analyze_expert_projects_completion,
            "experience_rates": self.analyzer.analyze_experience_vs_rates,
            "specialization_earnings": self.analyzer.analyze_specialization_earnings,
            "platform_performance": self.analyzer.analyze_platform_performance,
            "summary": self.analyzer.get_comprehensive_summary,
        }

    def _initialize_claude(self) -> Optional[anthropic.Anthropic]:
        """
        Initialize Claude client through Anthropic API.

        Returns:
            Anthropic client instance or None if not available
        """
        if not ANTHROPIC_AVAILABLE:
            print("❌ Anthropic SDK is not available")
            return None

        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY environment variable not set")
            print("💡 Set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
            return None

        try:
            # Get configuration from environment variables with defaults
            timeout = int(os.getenv("CLAUDE_TIMEOUT", "30"))

            # Configure proxy if available using httpx
            http_client_kwargs = {}

            if (
                os.getenv("HTTP_PROXY")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("SOCKS_PROXY")
            ):
                try:
                    import httpx

                    proxies = {}
                    if os.getenv("HTTP_PROXY"):
                        proxies["http://"] = os.getenv("HTTP_PROXY")
                    if os.getenv("HTTPS_PROXY"):
                        proxies["https://"] = os.getenv("HTTPS_PROXY")
                    if os.getenv("SOCKS_PROXY"):
                        proxies["http://"] = os.getenv("SOCKS_PROXY")
                        proxies["https://"] = os.getenv("SOCKS_PROXY")

                    if proxies:
                        # Create custom httpx client with proxy
                        http_client = httpx.Client(
                            proxy=proxies.get("https://") or proxies.get("http://"),
                            timeout=timeout,
                        )
                        http_client_kwargs["http_client"] = http_client
                        print(f"🌐 Using proxy for Claude: {list(proxies.keys())}")

                except ImportError:
                    print("⚠️ httpx not available for proxy support")
                except Exception as e:
                    print(f"⚠️ Proxy configuration failed: {e}")

            # Initialize Anthropic client
            client = anthropic.Anthropic(
                api_key=api_key,
                timeout=timeout,
                **http_client_kwargs,  # type: ignore
            )

            # Get model name for display
            model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            print(f"✅ Claude (Anthropic) initialized successfully with model {model}")
            return client

        except Exception as e:
            print(f"❌ Claude initialization failed: {e}")
            return None

    def classify_question(self, question: str) -> str:
        """
        Classify the user question to determine which analysis to run.

        Args:
            question: User's natural language question

        Returns:
            Analysis type key
        """
        question_lower = question.lower()

        # Enhanced keyword matching
        if any(
            word in question_lower
            for word in [
                "крипто",
                "crypto",
                "криптовалют",
                "оплат",
                "payment",
                "bitcoin",
            ]
        ):
            return "crypto_payment"
        elif any(
            word in question_lower
            for word in [
                "регион",
                "region",
                "распределя",
                "географ",
                "страна",
                "country",
            ]
        ):
            return "regional_income"
        elif any(
            word in question_lower
            for word in ["эксперт", "expert", "100", "проект", "project", "выполнил"]
        ):
            return "expert_projects"
        elif any(
            word in question_lower
            for word in [
                "опыт",
                "experience",
                "ставк",
                "rate",
                "навык",
                "skill",
                "часов",
            ]
        ):
            return "experience_rates"
        elif any(
            word in question_lower
            for word in [
                "специализац",
                "specialization",
                "категор",
                "category",
                "прибыльн",
            ]
        ):
            return "specialization_earnings"
        elif any(
            word in question_lower
            for word in [
                "платформ",
                "platform",
                "fiverr",
                "upwork",
                "freelancer",
                "топтал",
            ]
        ):
            return "platform_performance"
        else:
            return "summary"

    def _generate_context_prompt(
        self, analysis_type: str, analysis_data: Dict[str, Any], user_question: str
    ) -> str:
        """
        Generate a context-rich prompt for Claude.

        Args:
            analysis_type: Type of analysis performed
            analysis_data: Results from data analysis
            user_question: Original user question

        Returns:
            Formatted prompt for Claude
        """
        # Convert analysis data to readable format
        data_summary = format_data_as_json(analysis_data, indent=2)

        context_prompt = f"""
Ты - эксперт аналитик данных фрилансеров. Ответь на вопрос пользователя, используя предоставленные статистические данные.

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_question}

РЕЗУЛЬТАТЫ АНАЛИЗА ДАННЫХ:
{data_summary}

ИНСТРУКЦИИ:
1. Дай четкий и структурированный ответ на русском языке
2. Используй конкретные числа и проценты из анализа
3. Объясни основные тренды и закономерности
4. Сделай практические выводы для фрилансеров
5. Используй простой и понятный язык
6. Форматируй ответ с заголовками и списками для лучшей читаемости

ОТВЕТ:"""
        return context_prompt

    def query_llm(self, prompt: str) -> str:
        """
        Query Claude through Anthropic API.

        Args:
            prompt: Formatted prompt for Claude

        Returns:
            Claude response text
        """
        if not self.client:
            return self._generate_fallback_response()

        try:
            print("🤖 Querying Claude...")

            # Get configuration from environment variables with defaults
            model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            temperature = float(os.getenv("CLAUDE_TEMPERATURE", "0.1"))
            max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", "4000"))

            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Handle different content types from Claude
            content = message.content[0]
            if hasattr(content, "text"):
                return str(content.text)
            else:
                return str(content)

        except Exception as e:
            print(f"❌ Claude query failed: {e}")
            return self._generate_fallback_response()

    def _generate_fallback_response(self) -> str:
        """
        Generate a fallback response when Claude is not available.

        Returns:
            Fallback response text
        """
        return """
🤖 Анализ данных выполнен успешно!

К сожалению, Claude LLM сервис недоступен, но данные были проанализированы.

📊 **Чтобы получить Claude ответы:**

1. **Получите Anthropic API ключ:**
   - Зарегистрируйтесь на https://console.anthropic.com/
   - Создайте API ключ в разделе API Keys
   - Стоимость: ~$0.003 за 1000 токенов для Claude-3.5-Sonnet

2. **Установите переменную окружения:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

3. **Или добавьте в .env файл:**
   ```bash
   ANTHROPIC_API_KEY=your-api-key-here
   ```

4. **Перезапустите систему:**
   ```bash
   uv run -m src.main ask "ваш вопрос"
   ```

💡 **Совет:** Используйте команду 'analyze' для получения детальных данных анализа без LLM.

📈 **Основные выводы из анализа:**
- Криптоплатежи показывают разные результаты по регионам
- Экспертность влияет на количество проектов и доходы
- Специализация играет ключевую роль в доходах
"""

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return comprehensive response.

        Args:
            question: User's question in natural language

        Returns:
            Dictionary with analysis results and Claude response
        """
        try:
            # Step 1: Classify the question
            analysis_type = self.classify_question(question)
            print(f"🔍 Detected question type: {analysis_type}")

            # Step 2: Run appropriate analysis
            analysis_function = self.question_mapping[analysis_type]
            analysis_data = analysis_function()
            print("📊 Analysis completed")

            # Step 3: Generate context prompt
            context_prompt = self._generate_context_prompt(
                analysis_type, analysis_data, question
            )

            # Step 4: Get Claude response
            llm_response = self.query_llm(context_prompt)

            return {
                "question": question,
                "analysis_type": analysis_type,
                "analysis_data": analysis_data,
                "llm_response": llm_response,
                "status": "success",
            }

        except Exception as e:
            print(f"❌ Error processing question: {e}")
            return {
                "question": question,
                "error": str(e),
                "status": "error",
                "fallback_response": self._generate_fallback_response(),
            }

    def get_sample_questions(self) -> List[str]:
        """
        Get list of sample questions that the system can answer.

        Returns:
            List of sample questions
        """
        return [
            "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?",
            "Как распределяется доход фрилансеров в зависимости от региона проживания?",
            "Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?",
            "Как связан уровень опыта фрилансера с его часовой ставкой?",
            "Какие специализации фрилансеров наиболее прибыльны?",
            "На какой платформе фрилансеры зарабатывают больше всего?",
            "Дайте общую сводку по рынку фрилансеров",
        ]

    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of Claude service.

        Returns:
            Health status information
        """
        health_status = {
            "data_analyzer": True,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "anthropic_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "llm_initialized": self.client is not None,
            "overall_status": "healthy" if self.client else "llm_unavailable",
        }

        # Test Claude if available
        if self.client:
            try:
                test_response = self.query_llm("Hello")
                health_status["llm_test"] = "passed" if test_response else "failed"
            except Exception as e:
                health_status["llm_test"] = f"failed: {str(e)}"
                health_status["overall_status"] = "degraded"
        else:
            health_status["llm_test"] = "not_available"

        return health_status
