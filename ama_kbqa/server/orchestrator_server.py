from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
import os
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ConfigDict
from fastmcp import FastMCP, Context
from loguru import logger
from ama_kbqa.config import (
    assert_provider_api_key_present,
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
# NOTE: score_threshold is a shared config knob, also used by kqapro_server's
# entity search. Here it only filters which Qdrant hits appear as routing
# evidence; the routing decision itself is made by the orchestrator's LLM.
SCORE_THRESHOLD = get_score_threshold()


# --- 1. Define a Context Class for Type Safety ---

class AppContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Qdrant is optional: the server boots in a degraded mode (the probing
    # tool returns an evidence-free payload and the router LLM decides from
    # the question's domain alone) when the vector DB is unreachable,
    # instead of crashing the whole MCP server on startup.
    qdrant: Optional[QdrantClient]
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

    # Fail fast (and visibly in this subprocess's logs) if no provider key is
    # set, instead of dying later in get_chat_client() with the parent only
    # seeing an opaque "Connection closed".
    assert_provider_api_key_present()

    qdrant: Optional[QdrantClient] = None
    try:
        # The chat client is required: without it we can neither extract
        # semantics nor embed terms, so the orchestrator cannot route at
        # all. A failure here is fatal and re-raised below.
        # (Note: This assumes chat and embedding providers are the same)
        openai = get_chat_client()

        # Qdrant is best-effort. If it is unreachable we still boot so the
        # MCP server stays alive; analyze_query_recommend_db then degrades
        # to a KQAPro recommendation instead of crashing on startup with an
        # opaque "Connection closed" seen by the client.
        try:
            qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            qdrant.get_collections()  # Quick connectivity check
            logger.info("Connected to Qdrant.")
        except Exception as e:
            logger.warning(
                f"Qdrant unreachable at {QDRANT_HOST}:{QDRANT_PORT} ({e}); "
                "starting in degraded mode (routing will default to KQAPro)."
            )
            if qdrant is not None:
                try:
                    qdrant.close()
                except Exception:
                    pass
            qdrant = None

        # Yield the context so tools can access it
        yield AppContext(qdrant=qdrant, openai=openai)

    except Exception as e:
        logger.error(f"Something went wrong during startup: {e}")
        raise  # Re-raise to properly signal startup failure

    finally:
        # Cleanup code (runs on shutdown)
        logger.info("Shutting down: Closing connections...")
        if qdrant is not None:
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
                hits = qdrant.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=TOP_N,
                    with_payload=True,
                    score_threshold=SCORE_THRESHOLD
                ).points
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
    Probes both knowledge graphs (KQAPro and SciQA) with the entities of a
    natural language question and returns the raw linking evidence so the
    caller can decide which specialist agent to route to.

    It performs the following steps internally:
    1. Extracts semantic entities (NER).
    2. Embeds these entities into vectors.
    3. Searches BOTH Qdrant collections for matches.

    Args:
        question: The natural language question to analyze. Pass the user's
            question verbatim.
        context: The FastMCP request context containing the active database connections.

    Returns:
        A JSON string containing:
        - 'semantics': The extracted subject/predicate/objects.
        - 'kg_evidence': Per knowledge graph ('kqapro', 'sciqa'): how many
          probed terms matched, the average match score, and the best match
          per term (id, label, score). Inspect the matched labels; a high
          score on a semantically wrong entity is not real evidence.
        - 'degraded': true when no evidence could be gathered (vector DB
          down or entity extraction failed). Decide from the question's
          domain alone in that case.
        - 'note': Optional detail about why evidence is missing.
    """
    import time

    # 1. Get Context
    app_context: AppContext = context.request_context.lifespan_context

    logger.info(f"--- [Master Tool] Processing: '{question}' ---")
    start_time = time.time()

    # Degraded mode: the vector DB is unavailable, so no entity linking is
    # possible. Skip the (now pointless) extraction + embedding LLM calls
    # and return an evidence-free payload; the orchestrator's router LLM
    # then decides from the question's domain alone.
    if app_context.qdrant is None:
        logger.warning("Qdrant unavailable; returning evidence-free degraded payload.")
        return json.dumps({
            "semantics": {},
            "kg_evidence": {},
            "degraded": True,
            "note": "Vector database (Qdrant) is unavailable; no entity-linking "
                    "evidence could be gathered. Decide from the question's "
                    "domain alone.",
        }, indent=2, ensure_ascii=False)

    # 2. Extract
    extract_start = time.time()
    semantics = extract_semantics(app_context.openai, question)
    logger.debug(f"[TIMING] Semantic extraction took {time.time() - extract_start:.2f}s")

    # Check for errors in semantic extraction. Still return the evidence
    # shape (not a bare error) so the router LLM can fall back to deciding
    # from the question's domain instead of silently defaulting.
    if not semantics or "error" in semantics:
        error_detail = semantics.get("error", "Unknown error") if semantics else "No response from LLM"
        logger.error(f"Semantic extraction failed: {error_detail}")
        return json.dumps({
            "semantics": {},
            "kg_evidence": {},
            "degraded": True,
            "note": f"Entity extraction failed ({error_detail}); no entity-linking "
                    "evidence could be gathered. Decide from the question's "
                    "domain alone.",
        }, indent=2, ensure_ascii=False)

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

    # 5. Summarize the linking evidence per knowledge graph. No verdict is
    # computed here: the orchestrator's router LLM weighs this evidence
    # against the agents' domain descriptions. The old collapsed
    # recommendation was structurally biased toward KQAPro (its hardcoded
    # > 0.7 gate also disagreed with the configured score_threshold of 0.6,
    # so 0.6-0.7 hits counted as entities yet dragged the average below the
    # gate) and hid the matched labels needed to spot spurious matches.
    terms_probed = sum(1 for v in vectors_map.values() if v)
    kg_evidence = {}

    for kg_name, kg_results in search_results.items():
        matches = {}
        sum_scores = 0.0

        for term, candidates in kg_results.items():
            if candidates:
                best_match = candidates[0]
                sum_scores += best_match['score']
                matches[term] = {
                    "id": best_match['id'],
                    "label": best_match['label'],
                    "score": best_match['score']
                }

        kg_evidence[kg_name] = {
            "terms_probed": terms_probed,
            "terms_matched": len(matches),
            "avg_score": round(sum_scores / len(matches), 4) if matches else 0.0,
            "matches": matches
        }

    # 6. Construct Final Output
    final_output = {
        "semantics": semantics,
        "kg_evidence": kg_evidence,
        "degraded": False
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
