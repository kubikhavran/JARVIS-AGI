import pytest
import httpx
from pytest_httpx import HTTPXMock
from brain.ollama_client import OllamaClient

@pytest.mark.asyncio
async def test_ollama_chat_returns_string(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": "Hello!"}},
    )
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "Hello!"

@pytest.mark.asyncio
async def test_ollama_chat_timeout_raises(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.TimeoutException("timeout"))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(httpx.TimeoutException):
        await client.chat([{"role": "user", "content": "hi"}])

@pytest.mark.asyncio
async def test_gemini_client_no_key_raises():
    from brain.gemini_client import GeminiClient
    client = GeminiClient(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        await client.chat([{"role": "user", "content": "hi"}])

@pytest.mark.asyncio
async def test_router_returns_skill_and_params(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.base import BaseSkill

    class WeatherSkill(BaseSkill):
        name = "weather"
        description = "Get weather for a location"
        parameters = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        async def execute(self, **kwargs) -> str:
            return "sunny"

    registry = SkillRegistry()
    registry.register(WeatherSkill())

    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": '{"skill": "weather", "params": {"location": "Prague"}}'}},
    )
    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("What's the weather in Prague?", context=[])

    assert skill_name == "weather"
    assert params["location"] == "Prague"

@pytest.mark.asyncio
async def test_router_unknown_skill_falls_back_to_general_chat(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.general_chat import GeneralChatSkill

    registry = SkillRegistry()
    registry.register(GeneralChatSkill(ollama_url="http://localhost:11434", model="qwen3:8b"))

    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": '{"skill": "nonexistent_skill", "params": {}}'}},
    )
    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("something random", context=[])

    assert skill_name == "general_chat"

@pytest.mark.asyncio
async def test_router_retries_on_bad_json(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.general_chat import GeneralChatSkill

    registry = SkillRegistry()
    registry.register(GeneralChatSkill(ollama_url="http://localhost:11434", model="qwen3:8b"))

    # First response is invalid JSON, second is valid
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": "oops not json"}},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": '{"skill": "general_chat", "params": {"message": "hi"}}'}},
    )

    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("hi", context=[])
    assert skill_name == "general_chat"
