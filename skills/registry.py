from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
from skills.base import BaseSkill

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        return self._skills[name]

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._skills.values()
        ]

    def auto_discover(self) -> None:
        skills_path = Path(__file__).parent
        for finder, module_name, _ in pkgutil.iter_modules([str(skills_path)]):
            if module_name in ("base", "registry", "__init__"):
                continue
            module = importlib.import_module(f"skills.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                try:
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseSkill)
                        and attr is not BaseSkill
                    ):
                        self.register(attr())
                except TypeError:
                    pass
