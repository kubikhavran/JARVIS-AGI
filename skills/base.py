from abc import ABC, abstractmethod

class BaseSkill(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    async def execute(self, **kwargs) -> str: ...
