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
