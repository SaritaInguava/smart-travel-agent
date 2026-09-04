from langchain_mcp_adapters.client import MultiServerMCPClient

from smart_travel_agent.config import get_expedia_rapidapi_key

EXPEDIA_RAPIDAPI_HOST = "expedia-api-expedia-data-scraper.p.rapidapi.com"


def _expedia_mcp_config() -> dict:
    return {
        "expedia": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "mcp-remote",
                "https://mcp.rapidapi.com",
                "--header",
                f"x-api-host: {EXPEDIA_RAPIDAPI_HOST}",
                "--header",
                f"x-api-key: {get_expedia_rapidapi_key()}",
            ],
        }
    }


async def get_expedia_tools() -> list:
    client = MultiServerMCPClient(_expedia_mcp_config())
    return await client.get_tools()
