import pytest

from dbmind_ai import llm_factory
from dbmind_ai.config import Settings


def test_build_chat_model_uses_provider_prefixed_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected_model = object()

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(llm_factory, "init_chat_model", fake_init_chat_model)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test:test@localhost/test",
        llm_model="openai:o3",
        llm_max_retries=4,
    )

    model = llm_factory.build_chat_model(settings)

    assert model is expected_model
    assert captured == {
        "model": "openai:o3",
        "model_provider": None,
        "max_retries": 4,
    }


def test_build_chat_model_passes_explicit_provider_and_temperature(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_factory, "init_chat_model", fake_init_chat_model)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test:test@localhost/test",
        llm_model="deployment-name",
        llm_provider="azure_openai",
        llm_temperature=0,
    )

    llm_factory.build_chat_model(settings)

    assert captured["model_provider"] == "azure_openai"
    assert captured["temperature"] == 0


def test_build_chat_model_explains_missing_provider_package(monkeypatch) -> None:
    def missing_integration(**kwargs):
        raise ImportError("missing integration")

    monkeypatch.setattr(llm_factory, "init_chat_model", missing_integration)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://test:test@localhost/test",
        llm_model="anthropic:claude-example",
    )

    with pytest.raises(RuntimeError, match="anthropic"):
        llm_factory.build_chat_model(settings)
