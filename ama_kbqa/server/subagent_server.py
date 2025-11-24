from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP(name="kbqa_tools", json_response=True)
REPO_ROOT = Path(__file__).resolve().parents[2]

@mcp.tool()
def searchInfo(question: str) -> str:
    """Nimmt Frage entgegen und sucht die Antwort in Knowledge Bases."""
    return "Frage konnte nicht beantwortet werden. Selbst beantworten bitte."


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()