from mcp.server.fastmcp import FastMCP
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
import os
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ConfigDict
from fastmcp import Context
from loguru import logger
from ama_kbqa.config import (
    get_chat_client,
    get_chat_model_name,
    get_provider_preferences
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Configure logger
log_dir = REPO_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "orchestrator_server.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# 1. Load environment variables
load_dotenv(find_dotenv())

# Constants TODO: Get config parameters from config.toml instead of here
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
YOUR_SITE_URL = "https://my-mcp-server.local"
YOUR_SITE_NAME = "KBQA MCP Tool"

# Note: Check OpenRouter for the exact model ID.
# As of now, common IDs are 'alibaba/gte-qwen2-7b-instruct' or similar.
# I kept your requested ID, but if it fails, check the OpenRouter model list.
EMBEDDING_MODEL_ID = "qwen/qwen3-embedding-8b"
CHAT_MODEL_ID = "arcee-ai/trinity-mini"

# Qdrant Config
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "wikidata_entities"  # Ensure this exists in your Qdrant
TOP_N = 3                             # Candidates to retrieve per word
SCORE_THRESHOLD = 0.70                # Minimum similarity (Cosine)


# --- 1. Define a Context Class for Type Safety ---

class AppContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    qdrant: QdrantClient
    openai: OpenAI


# --- 2. Define the Lifespan Manager ---

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manages the lifecycle of the server.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    logger.info("Starting up: Connecting to Qdrant & OpenAI...")

    try:
        # Initialize Clients
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is missing in environment variables.")

        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Quick connectivity check
        qdrant.get_collections()

        openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Yield the context so tools can access it
        yield AppContext(qdrant=qdrant, openai=openai)

    except Exception as e:
        logger.error(f"Something went wrong: {e}")

    finally:
        # Cleanup code (runs on shutdown)
        logger.info("🔌 Shutting down: Closing connections...")
        qdrant.close()
        sys.exit(1)


# --- 3. Initialize FastMCP with Lifespan ---
mcp = FastMCP(name="orchestrator_tools", lifespan=server_lifespan, json_response=True)


# --- 4. Helper Functions (now needs the client passed in) ---

def extract_semantics(client: OpenAI, question: str) -> dict:
    """
    Internal: Extracts relations and ALL relevant entities/concepts from a question.

    Args:
        client: The OpenAI client instance.
        question: The sentence to analyze (e.g., "What is the planet closest to Earth?").

    Returns:
        A JSON string containing 'subject', 'predicate', and a list of 'objects'.
    """

    # The system prompt enforcing the structure
    system_prompt = (
        "You are an expert in Named Entity Recognition (NER) and Relation Extraction. "
        "Your task is to decompose a question into its semantic components so they can be used for a database query.\n"
        "Rules:\n"
        "1. 'subject': The question word (e.g., Who, What) or the acting noun (without articles).\n"
        "2. 'predicate': The main verb or the relation.\n"
        "3. 'objects': A LIST of all other semantically important terms. This includes:\n"
        "   - All nouns and proper names (e.g., 'Earth', 'USA').\n"
        "   - Important adjectives/superlatives defining a property (e.g., 'closest', 'first', 'most expensive').\n"
        "   - Split compound terms if necessary.\n"
        "   - Remove stop words, articles (the, a), and prepositions (of, in, at).\n"
        "Reply ONLY with a valid JSON object in this format: "
        '{"subject": "...", "predicate": "...", "objects": ["...", "..."]}'
    )

    logger.info(f"Analyzing: '{question}' ...")

    try:
        # Build API call parameters
        call_params = {
            "model": CHAT_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "response_format": {"type": "json_object"}
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}
            logger.debug(f"Using provider preferences: {provider_prefs}")

        completion = client.chat.completions.create(**call_params)

        # result_content = completion.choices[0].message.content
        return json.loads(completion.choices[0].message.content)
        # Validate JSON before returning
       # json_check = json.loads(result_content)

        # return json.dumps(json_check, ensure_ascii=False)

    except Exception as e:
        # return json.dumps({"error": f"Error while analyzing: {str(e)}"})
        logger.error(f"[_extract_semantics] Error: {e}")
        return {}


def generate_embeddings(client: OpenAI, terms: list[str]) -> dict:
    """
    Internal: Generates vector embeddings for a list of terms using the Qwen model via OpenRouter.

    Args:
        client: The OpenAI client instance.
        terms: A list of strings to embed (e.g., ["Earth", "Planet", "closest"]).

    Returns:
        A JSON string mapping each term to its vector embedding.
        Example: {"Earth": [0.12, ...], "Planet": [0.99, ...]}
    """

    # Input cleaning
    clean_texts = []
    if not terms:
        return json.dumps({})

    for t in terms:
        if isinstance(t, str):
            clean_texts.append(t.replace("\n", " "))
        elif isinstance(t, list):
            # If a list was accidentally passed inside the list
            joined = " ".join([str(x) for x in t])
            clean_texts.append(joined)
        else:
            clean_texts.append(str(t))

    logger.info(f"Generating embeddings for {len(clean_texts)} terms...")

    results = {}
    logger.debug(f"Terms to embed: {clean_texts}")

    # Try batch processing first (much faster)
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_ID,
            input=clean_texts  # Pass entire list at once
        )

        # Map results back to terms
        for i, text in enumerate(clean_texts):
            results[text] = response.data[i].embedding

    except Exception as e:
        logger.warning(f"Batch embedding failed ({e}), falling back to sequential processing...")

        # Fallback: Process sequentially if batch fails
        for text in clean_texts:
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL_ID,
                    input=text
                )
                results[text] = response.data[0].embedding
            except Exception as e:
                logger.error(f"Error embedding '{text}': {e}")
                results[text] = None

    return results


def _search_qdrant(qdrant: QdrantClient, vectors_map: dict) -> dict:
    """
    INTERNAL: Searches Qdrant.

    Args:
        qdrant: The Qdrant client instance.
        vectors_map: Dictionary mapping terms to their vector embeddings.

    Returns a Dictionary: {"term": [{"id": "...", "score": 0.9}, ...]}
    """
    search_results = {}

    for word, vector in vectors_map.items():
        if not vector:
            continue
        try:
            hits = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=TOP_N,
                with_payload=True,
                score_threshold=SCORE_THRESHOLD
            )
            candidates = []
            for hit in hits:
                candidates.append({
                    "id": hit.payload.get("id", "unknown"),
                    "label": hit.payload.get("label", "unknown"),
                    "score": round(hit.score, 4)
                })
            search_results[word] = candidates
        except Exception as e:
            logger.error(f"[_search_qdrant] Error for '{word}': {e}")
            search_results[word] = []

    return search_results


@mcp.tool()
def analyze_query_recommend_db(question: str, context: Context) -> str:
    """
    Analyzes a natural language question and recommends the best database strategy.

    It performs the following steps internally:
    1. Extracts semantic entities (NER).
    2. Embeds these entities into vectors.
    3. Checks the Vector Database (Qdrant) for matches.

    Args:
        question: The natural language question to analyze.
        context: The FastMCP request context containing the active database connections.

    Returns:
        A JSON string containing:
        - 'recommendation': Which DB to use ('vector_db' or 'text_search').
        - 'confidence': Average match score.
        - 'linked_entities': The identified IDs and Labels found in the DB.
    """
    import time

    # 1. Get Context
    app_context: AppContext = context.request_context.lifespan_context

    logger.info(f"--- [Master Tool] Processing: '{question}' ---")
    start_time = time.time()

    # 2. Extract
    extract_start = time.time()
    semantics = extract_semantics(app_context.openai, question)
    logger.debug(f"[TIMING] Semantic extraction took {time.time() - extract_start:.2f}s")
    if not semantics:
        return json.dumps({"error": "Failed to extract semantics."})

    # Prepare list for embedding
    terms_to_embed = []
    if semantics.get("subject"):
        terms_to_embed.append(semantics["subject"])
    if semantics.get("predicate"):
        terms_to_embed.append(semantics["predicate"])
    for obj in semantics.get("objects", []):
        terms_to_embed.append(obj)

    # 3. Embed
    embed_start = time.time()
    vectors_map = generate_embeddings(app_context.openai, terms_to_embed)
    logger.debug(f"[TIMING] Embedding generation took {time.time() - embed_start:.2f}s")

    # 4. Search / Validate
    search_start = time.time()
    search_results = _search_qdrant(app_context.qdrant, vectors_map)
    logger.debug(f"[TIMING] Qdrant search took {time.time() - search_start:.2f}s")

    # 5. LOGIC: Decide Recommendation
    # Calculate metrics to decide if we have enough "good" hits
    total_matches = 0
    sum_scores = 0
    found_entities = {}  # To store only the best match per term

    for term, candidates in search_results.items():
        if candidates:
            best_match = candidates[0]  # Take the top 1
            total_matches += 1
            sum_scores += best_match['score']

            # Structure specifically for the orchestrator
            found_entities[term] = {
                "db_id": best_match['id'],
                "db_label": best_match['label'],
                "confidence": best_match['score']
            }

    # Calculate average confidence of found items
    avg_confidence = (sum_scores / total_matches) if total_matches > 0 else 0.0

    # --- DECISION RULE ---
    # If we found at least one solid entity (e.g., Subject or Object) with high confidence,
    # we recommend the structured Vector/Graph DB.
    # Otherwise, we recommend a fallback (like text search or 'unknown').

    if total_matches >= 1 and avg_confidence > 0.7:
        recommendation = "take KQAPro for it "
        reasoning = f"Found {total_matches} entities with high confidence ({avg_confidence:.2f})."
    else:
        recommendation = "take KQAPro for it "  # Fallback
        reasoning = "No known entities found in the knowledge base."

    # 6. Construct Final Output
    final_output = {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "metrics": {
            "avg_confidence": avg_confidence,
            "entities_found_count": total_matches
        },
        "linked_entities": found_entities,  # The IDs the orchestrator needs for the next step
        "semantics": semantics    # Keep the extracted structure
    }

    logger.info(f"[TIMING] Total processing took {time.time() - start_time:.2f}s")
    return json.dumps(final_output, indent=2, ensure_ascii=False)


# @mcp.tool()
# def databaseSearch(question: str) -> str:
#     """Mache ein Preprocessing auf der aktuellen Anfrage, um zu sehen, auf welcher Knowledge Base gesucht werden soll."""
#     return "Nimm kqapro agent dafür"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
