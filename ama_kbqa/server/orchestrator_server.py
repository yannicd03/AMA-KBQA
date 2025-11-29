from mcp.server.fastmcp import FastMCP
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
import os
import json
from dotenv import load_dotenv, find_dotenv

mcp = FastMCP(name="orchestrator_tools", json_response=True)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 1. Load environment variables
load_dotenv(find_dotenv())

# Constants
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
YOUR_SITE_URL = "https://my-mcp-server.local"
YOUR_SITE_NAME = "KBQA MCP Tool"

# Note: Check OpenRouter for the exact model ID. 
# As of now, common IDs are 'alibaba/gte-qwen2-7b-instruct' or similar.
# I kept your requested ID, but if it fails, check the OpenRouter model list.
EMBEDDING_MODEL_ID = "qwen/qwen3-embedding-8b" 
CHAT_MODEL_ID = "openai/gpt-5"

#Qdrant Config
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "wikidata_entities" # Ensure this exists in your Qdrant
TOP_N = 3                             # Candidates to retrieve per word
SCORE_THRESHOLD = 0.70                # Minimum similarity (Cosine)
# Initialize FastMCP
#mcp = FastMCP("KBQA_Agent")

def get_client():
    """Helper to initialize the OpenAI client."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in environment variables.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def get_qdrant_client():
    """Helper to initialize the Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
#@mcp.tool()
def extract_semantics(question: str) -> dict:
    """
    Internal: Extracts relations and ALL relevant entities/concepts from a question.
    
    Args:
        question: The sentence to analyze (e.g., "What is the planet closest to Earth?").
        
    Returns:
        A JSON string containing 'subject', 'predicate', and a list of 'objects'.
    """
    client = get_client()
    
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
    
    print(f"Analyzing: '{question}' ...")

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model=CHAT_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"} 
        )

        #result_content = completion.choices[0].message.content
        return json.loads(completion.choices[0].message.content)
        # Validate JSON before returning
       # json_check = json.loads(result_content)
        
        #return json.dumps(json_check, ensure_ascii=False)

    except Exception as e:
        #return json.dumps({"error": f"Error while analyzing: {str(e)}"})
        print(f"[_extract_semantics] Error: {e}")
        return {}

#@mcp.tool()
def generate_embeddings(terms: list[str]) -> dict:
    """
    Internal: Generates vector embeddings for a list of terms using the Qwen model via OpenRouter.
    
    Args:
        terms: A list of strings to embed (e.g., ["Earth", "Planet", "closest"]).
        
    Returns:
        A JSON string mapping each term to its vector embedding.
        Example: {"Earth": [0.12, ...], "Planet": [0.99, ...]}
    """
    client = get_client()
    
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

    print(f"Generating embeddings for {len(clean_texts)} terms...")
    
    results = {}
    print(clean_texts)

    # Process sequentially (Qwen via OpenRouter often requires single inputs)
    for text in clean_texts:
        try:
            response = client.embeddings.create(
                model= EMBEDDING_MODEL_ID,
                input=text 
            )
            
            embedding = response.data[0].embedding
            results[text] = embedding 
            
        except Exception as e:
            print(f"Error embedding '{text}': {e}")
            results[text] = None
    
    return results

def _search_qdrant(vectors_map: dict) -> dict:
    """
    INTERNAL: Searches Qdrant.
    Returns a Dictionary: {"term": [{"id": "...", "score": 0.9}, ...]}
    """
    qdrant = get_qdrant_client()
    search_results = {}

    for word, vector in vectors_map.items():
        if not vector: continue
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
            print(f"[_search_qdrant] Error for '{word}': {e}")
            search_results[word] = []
            
    return search_results

@mcp.tool()
def analyze_query_recommend_db(question: str) -> str:
    """
    Analyzes a natural language question and recommends the best database strategy.
    
    It performs the following steps internally:
    1. Extracts semantic entities (NER).
    2. Embeds these entities into vectors.
    3. Checks the Vector Database (Qdrant) for matches.
    
    Returns:
        A JSON string containing:
        - 'recommendation': Which DB to use ('vector_db' or 'text_search').
        - 'confidence': Average match score.
        - 'linked_entities': The identified IDs and Labels found in the DB.
    """
    print(f"--- [Master Tool] Processing: '{question}' ---")
    
    # 1. Extract
    semantics = extract_semantics(question)
    if not semantics:
        return json.dumps({"error": "Failed to extract semantics."})
    
    # Prepare list for embedding
    terms_to_embed = []
    if semantics.get("subject"): terms_to_embed.append(semantics["subject"])
    if semantics.get("predicate"): terms_to_embed.append(semantics["predicate"])
    for obj in semantics.get("objects", []): terms_to_embed.append(obj)
    
    # 2. Embed
    vectors_map = generate_embeddings(terms_to_embed)
    
    # 3. Search / Validate
    search_results = _search_qdrant(vectors_map)
    
    # 4. LOGIC: Decide Recommendation
    # Calculate metrics to decide if we have enough "good" hits
    total_matches = 0
    sum_scores = 0
    found_entities = {} # To store only the best match per term

    for term, candidates in search_results.items():
        if candidates:
            best_match = candidates[0] # Take the top 1
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
        recommendation = "take KQAPro for it " # Fallback
        reasoning = "No known entities found in the knowledge base."

    # 5. Construct Final Output
    final_output = {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "metrics": {
            "avg_confidence": avg_confidence,
            "entities_found_count": total_matches
        },
        "linked_entities": found_entities, # The IDs the orchestrator needs for the next step
        "semantics": semantics    # Keep the extracted structure
    }
    
    return json.dumps(final_output, indent=2, ensure_ascii=False)


@mcp.tool()
def databaseSearch(question: str) -> str:
    """Mache ein Preprocessing auf der aktuellen Anfrage, um zu sehen, auf welcher Knowledge Base gesucht werden soll."""
    return "Nimm kqapro agent dafür"


def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()