from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP(name="orchestrator_tools", json_response=True)
REPO_ROOT = Path(__file__).resolve().parents[2]

@mcp.tool()
def databaseSearch(question: str) -> str:
    """Mache ein Preprocessing auf der aktuellen Anfrage, um zu sehen, auf welcher Knowledge Base gesucht werden soll."""
    return "Nimm kqapro agent dafür"


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()