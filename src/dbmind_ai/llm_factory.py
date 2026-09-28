"""Provider-neutral chat model construction."""

import os

from dotenv import dotenv_values, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from dbmind_ai.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured LangChain model without coupling agents to a provider."""

    load_dotenv()
    for name, value in dotenv_values().items():
        if value and not os.environ.get(name):
            os.environ[name] = value

    options: dict[str, object] = {"max_retries": settings.llm_max_retries}
    if settings.llm_temperature is not None:
        options["temperature"] = settings.llm_temperature

    try:
        return init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            **options,
        )
    except ImportError as exc:
        provider = settings.llm_provider or settings.llm_model.partition(":")[0]
        raise RuntimeError(
            f"The LangChain integration package for LLM provider '{provider}' is not installed."
        ) from exc
