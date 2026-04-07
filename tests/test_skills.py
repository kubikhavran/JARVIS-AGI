import pytest
from skills.base import BaseSkill
from skills.registry import SkillRegistry

class EchoSkill(BaseSkill):
    name = "echo"
    description = "Echoes back input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str, **kwargs) -> str:
        return text

@pytest.mark.asyncio
async def test_base_skill_execute():
    skill = EchoSkill()
    result = await skill.execute(text="hello")
    assert result == "hello"

def test_registry_register_and_get():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    assert "echo" in registry.skill_names
    skill = registry.get("echo")
    assert skill.name == "echo"

def test_registry_function_definitions():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    defs = registry.function_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "echo"
    assert "description" in defs[0]
    assert "parameters" in defs[0]

@pytest.mark.asyncio
async def test_system_info_returns_stats():
    from skills.system_info import SystemInfoSkill
    skill = SystemInfoSkill()
    result = await skill.execute()
    assert "CPU" in result
    assert "RAM" in result

@pytest.mark.asyncio
async def test_weather_skill_calls_wttr(httpx_mock):
    from skills.weather import WeatherSkill
    httpx_mock.add_response(text="Prague: ⛅ 12°C")
    skill = WeatherSkill()
    result = await skill.execute(location="Prague")
    assert "Prague" in result

@pytest.mark.asyncio
async def test_note_taker_save_and_list(tmp_path):
    from skills.note_taker import NoteTakerSkill
    skill = NoteTakerSkill(db_path=str(tmp_path / "test.db"))
    await skill.execute(action="save", content="Buy milk")
    result = await skill.execute(action="list")
    assert "Buy milk" in result

@pytest.mark.asyncio
async def test_web_search_returns_results():
    from skills.web_search import WebSearchSkill
    from unittest.mock import patch
    mock_results = [{"title": "Test", "body": "Result body", "href": "http://example.com"}]
    with patch("duckduckgo_search.DDGS.text", return_value=iter(mock_results)):
        skill = WebSearchSkill()
        result = await skill.execute(query="test query")
    assert "Result body" in result

@pytest.mark.asyncio
async def test_app_launcher_open():
    from skills.app_launcher import AppLauncherSkill
    from unittest.mock import patch
    with patch("subprocess.Popen"):
        skill = AppLauncherSkill()
        result = await skill.execute(app_name="notepad", action="open")
    assert "Opening" in result

@pytest.mark.asyncio
async def test_timer_alarm_creates_task():
    from skills.timer_alarm import TimerAlarmSkill
    skill = TimerAlarmSkill()
    result = await skill.execute(seconds=90, label="Test Timer")
    assert "1 minute" in result
    assert "30 seconds" in result

@pytest.mark.asyncio
async def test_file_manager_list(tmp_path):
    from skills.file_manager import FileManagerSkill
    (tmp_path / "file.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    skill = FileManagerSkill()
    result = await skill.execute(action="list", path=str(tmp_path))
    assert "file.txt" in result
    assert "subdir" in result

@pytest.mark.asyncio
async def test_screenshot_calls_pyautogui(tmp_path):
    from skills.screenshot import ScreenshotSkill
    from unittest.mock import patch
    out_path = str(tmp_path / "shot.png")
    with patch("pyautogui.screenshot") as mock_ss:
        skill = ScreenshotSkill()
        result = await skill.execute(save_path=out_path)
    mock_ss.assert_called_once_with(out_path)
    assert "shot.png" in result
