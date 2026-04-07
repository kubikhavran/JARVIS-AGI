from duckduckgo_search import DDGS
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "Search the web and return top results"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 3, **kwargs) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            if not results:
                return "No results found."
            lines = [f"{r['title']}: {r['body']}" for r in results]
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {e}"
