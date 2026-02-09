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
    get_embedding_model_name,
    get_provider_preferences,
    get_qdrant_host,
    get_qdrant_port,
    get_top_n,
    get_score_threshold,
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

# Get all configuration from centralized config

# Qdrant Config (from centralized config)
QDRANT_HOST = get_qdrant_host()
QDRANT_PORT = get_qdrant_port()
COLLECTION_KQAPRO = "kqapro-entities"
COLLECTION_SCIQA = "sciqa-entities"
TOP_N = get_top_n()
SCORE_THRESHOLD = get_score_threshold()


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
        # Initialize Clients using centralized config
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Quick connectivity check
        qdrant.get_collections()

        # Use centralized config to get the chat client
        # (Note: This assumes chat and embedding providers are the same)
        openai = get_chat_client()

        # Yield the context so tools can access it
        yield AppContext(qdrant=qdrant, openai=openai)

    except Exception as e:
        logger.error(f"Something went wrong during startup: {e}")
        raise  # Re-raise to properly signal startup failure

    finally:
        # Cleanup code (runs on shutdown)
        logger.info("Shutting down: Closing connections...")
        qdrant.close()


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
        # Build API call parameters using centralized config
        call_params = {
            "model": get_chat_model_name(),
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
        content = completion.choices[0].message.content
        if content:
            return json.loads(content)

        # If json_object mode returned empty, retry without it
        logger.warning("JSON mode returned empty, retrying without response_format")
        call_params.pop("response_format", None)
        completion = client.chat.completions.create(**call_params)
        content = completion.choices[0].message.content or ""

        # Try to extract JSON from text response
        import re
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            return json.loads(json_match.group())

        return {"error": f"Could not parse JSON from LLM response: {content[:200]}"}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[_extract_semantics] Error: {error_msg}")
        return {"error": error_msg}


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
            model=get_embedding_model_name(),
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
                    model=get_embedding_model_name(),
                    input=text
                )
                results[text] = response.data[0].embedding
            except Exception as e:
                logger.error(f"Error embedding '{text}': {e}")
                results[text] = None

    return results


def _search_qdrant(qdrant: QdrantClient, vectors_map: dict) -> dict:
    """
    INTERNAL: Searches both KQAPro and SciQA Qdrant collections.

    Args:
        qdrant: The Qdrant client instance.
        vectors_map: Dictionary mapping terms to their vector embeddings.

    Returns a Dictionary:
        {"kqapro": {"term": [{"id": ..., "score": ...}]},
         "sciqa":  {"term": [{"id": ..., "score": ...}]}}
    """
    results = {"kqapro": {}, "sciqa": {}}

    for collection_key, collection_name in [("kqapro", COLLECTION_KQAPRO), ("sciqa", COLLECTION_SCIQA)]:
        for word, vector in vectors_map.items():
            if not vector:
                continue
            try:
                hits = qdrant.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=TOP_N,
                    with_payload=True,
                    score_threshold=SCORE_THRESHOLD
                )
                candidates = []
                for hit in hits:
                    candidates.append({
                        "id": hit.payload.get("original_id", hit.payload.get("id", "unknown")),
                        "label": hit.payload.get("name", hit.payload.get("label", "unknown")),
                        "score": round(hit.score, 4)
                    })
                results[collection_key][word] = candidates
            except Exception as e:
                logger.error(f"[_search_qdrant] Error for '{word}' in {collection_name}: {e}")
                results[collection_key][word] = []

    return results


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

    # Check for errors in semantic extraction
    if not semantics or "error" in semantics:
        error_detail = semantics.get("error", "Unknown error") if semantics else "No response from LLM"
        logger.error(f"Semantic extraction failed: {error_detail}")
        return json.dumps({
            "error": "Failed to extract semantics",
            "details": error_detail,
            "suggestion": "Check your LLM configuration and API key in config.toml"
        })

    # Only embed the object terms (nouns/entities) for routing decisions.
    # Subject (Who/What) and predicate (verbs) match broadly in any collection.
    object_terms = semantics.get("objects", [])
    if not object_terms:
        # Fallback: use all terms if no objects extracted
        if semantics.get("subject"):
            object_terms.append(semantics["subject"])
        if semantics.get("predicate"):
            object_terms.append(semantics["predicate"])

    # 3. Embed
    embed_start = time.time()
    vectors_map = generate_embeddings(app_context.openai, object_terms)
    logger.debug(f"[TIMING] Embedding generation took {time.time() - embed_start:.2f}s")

    # 4. Search both KQAPro and SciQA collections
    search_start = time.time()
    search_results = _search_qdrant(app_context.qdrant, vectors_map)
    logger.debug(f"[TIMING] Qdrant search took {time.time() - search_start:.2f}s")

    # 5. Score each knowledge graph by its best matches
    kg_scores = {}
    kg_entities = {}

    for kg_name, kg_results in search_results.items():
        total_matches = 0
        sum_scores = 0
        found_entities = {}

        for term, candidates in kg_results.items():
            if candidates:
                best_match = candidates[0]
                total_matches += 1
                sum_scores += best_match['score']
                found_entities[term] = {
                    "db_id": best_match['id'],
                    "db_label": best_match['label'],
                    "confidence": best_match['score']
                }

        avg_confidence = (sum_scores / total_matches) if total_matches > 0 else 0.0
        kg_scores[kg_name] = {
            "avg_confidence": avg_confidence,
            "entities_found_count": total_matches
        }
        kg_entities[kg_name] = found_entities

    # 6. Pick the best knowledge graph (highest avg confidence on object terms).
    # On ties, prefer KQAPro (general domain) over SciQA (scientific niche).
    kg_priority = {"kqapro": 1, "sciqa": 0}
    best_kg = max(kg_scores, key=lambda k: (kg_scores[k]["avg_confidence"], kg_priority.get(k, 0)))
    best_metrics = kg_scores[best_kg]

    if best_kg == "sciqa" and best_metrics["entities_found_count"] >= 1 and best_metrics["avg_confidence"] > 0.7:
        recommendation = "Use SciQA"
        reasoning = f"Found {best_metrics['entities_found_count']} entities in SciQA with high confidence ({best_metrics['avg_confidence']:.2f})."
    elif best_metrics["entities_found_count"] >= 1 and best_metrics["avg_confidence"] > 0.7:
        recommendation = "Use KQAPro"
        reasoning = f"Found {best_metrics['entities_found_count']} entities in KQAPro with high confidence ({best_metrics['avg_confidence']:.2f})."
    else:
        recommendation = "Use KQAPro"
        reasoning = "No strong entity matches found; defaulting to KQAPro for a diverse Knowledge Basis."

    # 7. Construct Final Output
    final_output = {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "metrics": kg_scores,
        "linked_entities": kg_entities.get(best_kg, {}),
        "semantics": semantics
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
