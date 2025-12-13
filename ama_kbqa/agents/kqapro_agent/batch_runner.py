"""
Batch Runner for KQAPro Benchmark

This script processes a random sample of questions from the validation dataset
and saves detailed results including metadata for analysis.

Usage:
    python ama_kbqa/agents/kqapro_agent/batch_runner.py --n_questions 10 --seed 42 --postprocessing_mode sparql 
"""

from __future__ import annotations
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
import os
import sys
import asyncio
import json
import random
import time
import argparse
import toml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(override=True)

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
project_root = current_file.parents[3]  # Go up one more level to project root
sys.path.insert(0, str(ama_kbqa_root))


# ============================================================================
# CONFIGURATION
# ============================================================================

VALIDATION_DATASET_PATH = project_root / "db" / "datasets" / "kqapro" / "val.json"
BATCH_RESULTS_BASE_DIR = project_root / "batch_results"

# OpenRouter configuration for answer selection
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "arcee-ai/trinity-mini")

# Virtuoso SPARQL endpoint configuration
VIRTUOSO_ENDPOINT = "http://localhost:8890/sparql"

# SPARQL prefixes (matching kqapro_server.py)
SPARQL_PREFIXES = """
PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""

# Ensure batch_results directory exists
BATCH_RESULTS_BASE_DIR.mkdir(exist_ok=True)


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class AnswerJudgment(BaseModel):
    """Structured judgment of an answer's correctness and quality."""

    is_correct: bool = Field(
        description="Whether the predicted answer matches the gold answer (semantically equivalent)"
    )

    correctness_reasoning: str = Field(
        description="Detailed explanation of why the answer is correct or incorrect, comparing predicted vs. gold answer"
    )

    argumentation_quality: str = Field(
        description="Assessment of the agent's reasoning structure: Was it logical? Did it follow sound steps? Were there gaps or errors in reasoning?"
    )

    argumentation_score: int = Field(
        ge=1, le=5,
        description="Numeric score for argumentation quality (1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent)"
    )

    suggested_improvement: str = Field(
        description="Specific, actionable suggestion for how the agent could improve its approach for similar questions in the future"
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def export_fewshot_examples_from_judgments(
    results: List[Dict[str, Any]],
    output_dir: Path = None,
    min_argumentation_score: int = 4
) -> Dict[str, int]:
    """
    Export successful classifications as few-shot examples for future use.

    Extracts high-quality question classifications from batch results and saves them
    to the fewshot-examples directory, grouped by question type.

    Args:
        results: List of batch processing results with judgments
        output_dir: Directory to save examples (defaults to db/datasets/kqapro/fewshot-examples)
        min_argumentation_score: Minimum argumentation score to include (1-5)

    Returns:
        Dictionary mapping question types to number of examples exported
    """
    if output_dir is None:
        output_dir = project_root / "db" / "datasets" / "kqapro" / "fewshot-examples"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Group examples by question type
    examples_by_type = {}

    for result in results:
        # Only export correct answers with high argumentation scores
        if not result.get("accuracy", False):
            continue

        judgment = result.get("judgment")
        if not judgment:
            continue

        if judgment.get("argumentation_score", 0) < min_argumentation_score:
            continue

        qtype = result.get("qtype", "Unknown")
        if qtype == "Unknown":
            continue

        # Create the example entry
        example = {
            "question": result["question"],
            "reasoning": judgment.get("correctness_reasoning", ""),
            "correct_qtype": qtype,
            "lesson_learned": judgment.get("suggested_improvement", ""),
            "argumentation_score": judgment.get("argumentation_score", 0),
            "exported_at": datetime.now().isoformat()
        }

        if qtype not in examples_by_type:
            examples_by_type[qtype] = []

        examples_by_type[qtype].append(example)

    # Save examples to files
    export_counts = {}

    for qtype, examples in examples_by_type.items():
        example_file = output_dir / f"{qtype}.json"

        # Load existing examples
        existing_examples = []
        if example_file.exists():
            try:
                with open(example_file, "r", encoding="utf-8") as f:
                    existing_examples = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not load existing examples from {example_file}, overwriting")

        # Merge with new examples (avoid duplicates based on question text)
        existing_questions = {ex.get("question", "") for ex in existing_examples}
        new_examples = [ex for ex in examples if ex["question"] not in existing_questions]

        if new_examples:
            all_examples = existing_examples + new_examples

            # Sort by argumentation score (highest first)
            all_examples.sort(key=lambda x: x.get("argumentation_score", 0), reverse=True)

            # Limit to top 10 examples per type
            all_examples = all_examples[:10]

            # Save to file
            with open(example_file, "w", encoding="utf-8") as f:
                json.dump(all_examples, f, indent=2, ensure_ascii=False)

            export_counts[qtype] = len(new_examples)
            print(f"[OK] Exported {len(new_examples)} new examples for {qtype} (total: {len(all_examples)})")
        else:
            export_counts[qtype] = 0

    return export_counts


def load_judge_config() -> Dict[str, Any]:
    """
    Load judge configuration from config.toml.

    Returns:
        Dictionary with judge configuration
    """
    config_path = project_root / "config.toml"
    if not config_path.exists():
        print("[WARNING] config.toml not found, using defaults")
        return {
            "provider": "openrouter",
            "model": "minimax/minimax-m2",
            "temperature": 0.0
        }

    try:
        config = toml.load(config_path)
        postprocessing = config.get("postprocessing", {})

        # Get provider-specific settings
        provider = postprocessing.get("judge_provider", "openrouter")
        provider_config = config.get(provider, {})

        judge_model = postprocessing.get("judge_model")
        if not judge_model:
            # Fallback to chat_model from the provider
            judge_model = provider_config.get("chat_model", "minimax/minimax-m2")

        return {
            "provider": provider,
            "model": judge_model,
            "temperature": postprocessing.get("judge_temperature", 0.0),
            "base_url": provider_config.get("base_url", "https://openrouter.ai/api/v1"),
            "api_key_env": f"{provider.upper()}_API_KEY" if provider != "openrouter" else "OPENROUTER_API_KEY"
        }
    except Exception as e:
        print(f"[WARNING] Failed to load judge config: {e}, using defaults")
        return {
            "provider": "openrouter",
            "model": "minimax/minimax-m2",
            "temperature": 0.0,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY"
        }


def get_next_batch_folder() -> Path:
    """
    Find the next available batch folder (Batch001, Batch002, etc.)

    Returns:
        Path to the next batch folder
    """
    batch_num = 1
    while True:
        batch_folder = BATCH_RESULTS_BASE_DIR / f"Batch{batch_num:03d}"
        if not batch_folder.exists():
            batch_folder.mkdir(parents=True, exist_ok=True)
            return batch_folder
        batch_num += 1


def load_validation_dataset() -> List[Dict[str, Any]]:
    """
    Load the validation dataset from the JSON file.

    Returns:
        List of validation questions with metadata
    """
    with open(VALIDATION_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[OK] Loaded {len(data)} questions from validation dataset")
    return data


def sample_questions(
    data: List[Dict[str, Any]],
    n: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Randomly sample n questions from the dataset with a fixed seed.

    Args:
        data: Full validation dataset
        n: Number of questions to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled questions
    """
    random.seed(seed)
    n_sample = min(n, len(data))
    sampled = random.sample(data, n_sample)

    print(f"[OK] Sampled {n_sample} questions with seed={seed}")
    return sampled


def classify_question_type(question: str, program: List[Dict] = None) -> str:
    """
    Attempt to classify the question type based on the question text and program.

    Args:
        question: The question text
        program: The program structure (optional)

    Returns:
        Estimated question type
    """
    question_lower = question.lower()

    # Check program if available
    if program and len(program) > 0:
        last_function = program[-1].get("function", "")

        if last_function == "Count":
            return "Count"
        elif last_function in ["VerifyQuery", "Verify"]:
            return "Verify"
        elif last_function in ["SelectBetween", "SelectAmong"]:
            return "Select"
        elif last_function == "QueryAttr":
            return "Query"
        elif last_function == "QueryRelation":
            return "QueryRelation"
        elif last_function == "QueryAttrQualifier":
            return "QueryAttrQualifier"
        elif last_function == "QueryRelationQualifier":
            return "QueryRelationQualifier"

    # Fallback to question text analysis
    if question_lower.startswith("how many"):
        return "Count"
    elif question_lower.startswith(("is ", "does ", "did ", "was ", "were ", "are ")):
        return "Verify"
    elif question_lower.startswith(("what ", "which ", "who ", "when ", "where ", "whose ")):
        return "Query"

    return "Unknown"


def select_answer_from_choices(
    question: str,
    predicted_answer: str,
    choices: List[str],
    client: OpenAI
) -> str:
    """
    Use the LLM to select the most suitable answer from the available choices
    based on the question and the agent's predicted answer.

    This is a non-agentic, single-call approach that maps the verbose agent
    response to one of the predefined answer choices.

    Args:
        question: The original question
        predicted_answer: The verbose answer from the agent
        choices: List of available answer choices
        client: OpenAI client instance

    Returns:
        The selected answer from the choices list
    """
    # Filter out "unknown" choices to only show valid options
    valid_choices = [c for c in choices if c.lower() != "unknown"]

    # If no valid choices, return the first choice as fallback
    if not valid_choices:
        return choices[0] if choices else "unknown"

    # First, try simple string matching - look for choices in the predicted answer
    predicted_lower = predicted_answer.lower()

    # Exact match with emphasis markers (**, etc.)
    for choice in valid_choices:
        # Look for the choice emphasized in the answer (e.g., **choice** or "choice")
        if f"**{choice.lower()}**" in predicted_lower or f'"{choice.lower()}"' in predicted_lower:
            print(f"[MATCH] Found emphasized choice '{choice}' in predicted answer")
            return choice

    # Check for "yes" or "no" answers in boolean questions
    if "yes" in valid_choices and "no" in valid_choices:
        # Look for strong yes/no indicators
        if predicted_lower.startswith("yes") or " yes," in predicted_lower or "answer: yes" in predicted_lower or "answer is yes" in predicted_lower:
            print("[MATCH] Detected 'yes' answer from predicted text")
            return "yes"
        elif predicted_lower.startswith("no") or " no," in predicted_lower or "answer: no" in predicted_lower or "answer is no" in predicted_lower:
            print("[MATCH] Detected 'no' answer from predicted text")
            return "no"

    # Look for exact word matches (whole words only)
    import re
    for choice in valid_choices:
        # Use word boundaries to find exact matches
        pattern = r'\b' + re.escape(choice.lower()) + r'\b'
        if re.search(pattern, predicted_lower):
            print(f"[MATCH] Found exact word match for choice '{choice}' in predicted answer")
            return choice

    # Build the prompt for answer selection with improved instructions
    choices_numbered = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(valid_choices))

    prompt = f"""You are an answer extraction system. Your task is to identify which of the provided choices best matches the detailed answer.

Question: {question}

Detailed Answer: {predicted_answer}

Available Choices:
{choices_numbered}

CRITICAL INSTRUCTIONS:
1. Read the detailed answer carefully
2. Identify the main answer or conclusion in the detailed answer
3. Select the choice that EXACTLY matches this answer
4. If the answer says "yes", select "yes". If it says "no", select "no"
5. If a specific choice is mentioned or emphasized (in bold, quotes, etc.), select that choice
6. Respond with ONLY the number of your choice (1, 2, 3, etc.) OR the exact choice text

Your selection:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise answer selector. Respond with only the number or exact text of the matching choice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic selection
            max_tokens=100
        )

        selected = response.choices[0].message.content.strip()
        print(f"[DEBUG] LLM selected: '{selected}'")

        # Try to parse as a number first
        try:
            choice_num = int(selected.split('.')[0].strip())
            if 1 <= choice_num <= len(valid_choices):
                print(f"[MATCH] LLM selected choice {choice_num}: '{valid_choices[choice_num-1]}'")
                return valid_choices[choice_num - 1]
        except (ValueError, IndexError):
            pass

        # Verify the selected answer is in the valid choices (case-insensitive)
        selected_lower = selected.lower()
        for choice in valid_choices:
            if choice.lower() == selected_lower:
                print(f"[MATCH] LLM selected exact match: '{choice}'")
                return choice

        # If exact match not found, try to find partial match
        for choice in valid_choices:
            if choice.lower() in selected_lower or selected_lower in choice.lower():
                print(f"[MATCH] LLM selected partial match: '{choice}'")
                return choice

        # If no match found, use the first valid choice as fallback
        print(f"[WARNING] LLM selected '{selected}' which is not in choices. Using first choice: '{valid_choices[0]}'")
        return valid_choices[0]

    except Exception as e:
        print(f"[ERROR] Failed to select answer from choices: {e}")
        # Return first valid choice as fallback
        return valid_choices[0] if valid_choices else (choices[0] if choices else "unknown")


def synthesize_final_sparql(
    question: str,
    agent_messages: List[Any],
    client: OpenAI
) -> Optional[str]:
    """
    Synthesizes a single final SPARQL query from the agent's conversation history.

    This function analyzes all tool calls, scratchpad entries, and verified facts
    to construct a comprehensive SPARQL query that directly answers the question.

    Args:
        question: The original question
        agent_messages: The full message history from the agent
        client: OpenAI client instance

    Returns:
        A SPARQL query string, or None if synthesis fails
    """
    # Extract tool calls and results from message history
    tool_interactions = []
    discovered_triples = []  # Store actual triple patterns discovered

    for msg in agent_messages:
        # Handle both dict and object-based messages
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])
            tool_call_id = getattr(msg, "tool_call_id", None)

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_name = tc.get('function', {}).get('name')
                    tool_args = tc.get('function', {}).get('arguments')
                else:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments

                tool_interactions.append(f"Tool Call: {tool_name} with args {tool_args}")

        elif role == "tool" and content:
            tool_interactions.append(f"Tool Result: {content[:500]}")  # Truncate long results

            # Try to extract verified triple patterns from ExploreNeighborhood results
            try:
                if "verified_match" in content and "predicate_used" in content:
                    # Parse the JSON result to extract triple patterns
                    result_data = json.loads(content)
                    if result_data.get("verified_match"):
                        base_node = result_data.get("base_node", "")
                        predicate = result_data["verified_match"].get("predicate_used", "")
                        objects = result_data["verified_match"].get("objects", [])

                        if base_node and predicate:
                            # Store the verified triple pattern
                            for obj in objects[:3]:  # Limit to first 3 objects
                                obj_value = obj.get("value", "")
                                obj_type = obj.get("type", "")
                                if obj_type == "uri":
                                    discovered_triples.append(f"ex:{base_node} prop:{predicate} <{obj_value}>")
                                else:
                                    # It's a literal - store both the pattern and a sample value
                                    discovered_triples.append(
                                        f"ex:{base_node} prop:{predicate} ?value (found: '{obj_value}')")
            except:
                # If parsing fails, just continue
                pass

    # Build context from interactions
    context = "\n".join(tool_interactions) if tool_interactions else "No tool interactions recorded"

    # Add discovered triple patterns to context
    if discovered_triples:
        context += "\n\nVERIFIED TRIPLE PATTERNS (use these directly):\n" + "\n".join(discovered_triples)

    # Build the synthesis prompt
    prompt = f"""You are a SPARQL query synthesis expert for the KQAPro knowledge graph. Based on the question and the agent's exploration, generate a SINGLE, complete SPARQL query that directly answers the question.

Question: {question}

Agent's Exploration (Tool Calls and Results):
{context}

CRITICAL INSTRUCTIONS - PREFIX USAGE:
1. For ENTITIES (like Q217008, Q64, etc.), use the prefix "ex:" - Example: ex:Q217008
2. For PROPERTIES/RELATIONS (like P36, P1082, etc.), use the prefix "prop:" - Example: prop:P36
3. For ATTRIBUTES, use the prefix "attr:" - Example: attr:language
4. NEVER use "wd:", "wdt:", or other Wikidata prefixes - this is NOT Wikidata!
5. DO NOT include PREFIX declarations in your response - they will be added automatically

CRITICAL RDF/SPARQL RULES:
1. If you see "VERIFIED TRIPLE PATTERNS" above, USE THEM DIRECTLY in your query
2. NEVER try to query properties on literal values (strings, numbers, URLs)
3. If a property returns a literal (like a URL or string), you CANNOT query further properties on it
4. Only entities (things with URIs like ex:Q64) can have properties, NOT literals
5. If the answer is already in the verified triple patterns as a literal, just return that value directly

QUERY STRUCTURE:
1. Analyze the VERIFIED TRIPLE PATTERNS first - these are proven to exist in the database
2. Synthesize this information into ONE comprehensive SPARQL query
3. The query should directly answer the original question
4. Return ONLY the SPARQL query itself (SELECT/ASK/COUNT), no explanations
5. If counting, use COUNT(?var) or COUNT(DISTINCT ?var)
6. If verifying (yes/no question), use ASK query
7. Otherwise use SELECT to get the answer values

Example correct queries:
- Simple property: SELECT ?value WHERE {{ ex:Q64 prop:P1082 ?value . }}
- With filter: SELECT ?value WHERE {{ ex:Q64 prop:P856 ?value . FILTER(contains(?value, "blade")) }}
- Count: SELECT (COUNT(?value) AS ?count) WHERE {{ ex:Q64 prop:P150 ?value . }}

Your SPARQL query:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a SPARQL query expert. Generate precise, executable SPARQL queries based on conversation context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        query = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if query.startswith("```sparql") or query.startswith("```"):
            query = query.replace("```sparql", "").replace("```", "").strip()

        # Remove any PREFIX declarations the LLM might have added
        lines = query.split("\n")
        query_lines = [line for line in lines if not line.strip().startswith("PREFIX")]
        query = "\n".join(query_lines).strip()

        # Fix common prefix errors (Wikidata -> KQAPro)
        query = query.replace("wd:", "ex:")  # Entity prefix
        query = query.replace("wdt:", "prop:")  # Property prefix
        query = query.replace("wikibase:", "")  # Remove wikibase references
        query = query.replace("bd:", "")  # Remove blazegraph/wikidata specific prefixes

        # Fix schema.org or other common mistakes
        query = query.replace("schema:", "rdfs:")

        print(f"[SPARQL] Synthesized query:\n{query}")
        return query

    except Exception as e:
        print(f"[ERROR] Failed to synthesize SPARQL query: {e}")
        return None


def execute_sparql_postprocessing(
    question: str,
    agent_messages: List[Any],
    client: OpenAI,
    sparql_wrapper: SPARQLWrapper
) -> tuple[Optional[str], Optional[str]]:
    """
    Execute SPARQL-based postprocessing to get the final answer.

    Args:
        question: The original question
        agent_messages: The agent's message history
        client: OpenAI client for query synthesis
        sparql_wrapper: SPARQLWrapper instance for query execution

    Returns:
        Tuple of (final_answer, sparql_query_used)
    """
    # Step 1: Synthesize the SPARQL query
    query = synthesize_final_sparql(question, agent_messages, client)

    if not query:
        print("[ERROR] Could not synthesize SPARQL query")
        return None, None

    # Step 2: Add prefixes and execute
    full_query = f"{SPARQL_PREFIXES}\n{query}"

    try:
        sparql_wrapper.setQuery(full_query)
        results = sparql_wrapper.query().convert()

        # Step 3: Format the results based on query type
        if query.strip().upper().startswith("ASK"):
            # Boolean query
            answer = "yes" if results.get("boolean", False) else "no"
            print(f"[SPARQL] ASK query result: {answer}")
            return answer, query

        elif "COUNT" in query.upper():
            # Count query
            bindings = results.get("results", {}).get("bindings", [])
            if bindings and len(bindings) > 0:
                # Get the first variable's value (usually named ?count or similar)
                first_var = list(bindings[0].keys())[0]
                count_value = bindings[0][first_var]["value"]
                print(f"[SPARQL] COUNT query result: {count_value}")
                return str(count_value), query
            else:
                return "0", query

        else:
            # SELECT query
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
                # Extract all values from the first binding
                values = []
                for var_name, var_data in bindings[0].items():
                    values.append(var_data["value"])

                answer = ", ".join(values) if len(values) > 1 else values[0]
                print(f"[SPARQL] SELECT query result: {answer}")
                return answer, query
            else:
                print("[SPARQL] Query returned no results")
                return "unknown", query

    except Exception as e:
        print(f"[ERROR] SPARQL execution failed: {e}")
        return None, query


def execute_llm_judge_postprocessing(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    agent_messages: List[Any],
    client: OpenAI,
    model_name: str
) -> tuple[Optional[AnswerJudgment], bool]:
    """
    Use an LLM with structured output to judge answer correctness and quality.

    Args:
        question: The original question
        gold_answer: The correct answer from the dataset
        predicted_answer: The agent's predicted answer
        agent_messages: The agent's message history for reasoning analysis
        client: OpenAI client for LLM calls
        model_name: Model to use for judging

    Returns:
        Tuple of (judgment, accuracy)
    """
    # Extract the agent's reasoning from message history
    reasoning_steps = []
    for msg in agent_messages:
        # Handle both dict and object-based messages
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])

        if role == "assistant":
            if content and content.strip():
                reasoning_steps.append(f"Agent reasoning: {content[:500]}")
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get('function', {}).get('name')
                    else:
                        tool_name = tc.function.name
                    reasoning_steps.append(f"Tool used: {tool_name}")

    reasoning_context = "\n".join(reasoning_steps) if reasoning_steps else "No detailed reasoning available"

    # Build the judgment prompt
    prompt = f"""You are an expert evaluator for a knowledge base question answering system. Your task is to judge the quality of the agent's answer and reasoning process.

Question: {question}

Gold (Correct) Answer: {gold_answer}

Agent's Predicted Answer: {predicted_answer}

Agent's Reasoning Process:
{reasoning_context}

Evaluate the following:

1. **Correctness**: Does the predicted answer match the gold answer? Consider semantic equivalence, not just exact string matching. For example, "yes" and "Yes, it is true" should both be considered correct if the gold answer is "yes".

2. **Correctness Reasoning**: Explain in detail why you judged the answer as correct or incorrect. Compare the predicted answer to the gold answer.

3. **Argumentation Quality**: Analyze the agent's reasoning process. Was it:
   - Logical and coherent?
   - Following sound reasoning steps?
   - Using appropriate tools?
   - Making any logical errors or gaps?
   - Efficient or unnecessarily complex?

4. **Argumentation Score**: Rate the reasoning quality from 1-5:
   - 1: Very poor (major logical errors, nonsensical approach)
   - 2: Poor (significant gaps in reasoning, inefficient)
   - 3: Acceptable (reaches answer but with some issues)
   - 4: Good (solid reasoning with minor issues)
   - 5: Excellent (flawless, efficient, well-structured)

5. **Suggested Improvement**: Provide ONE specific, actionable suggestion for how the agent could improve its approach for this type of question in the future.

Provide your judgment in structured format."""

    try:
        # Use structured output with Pydantic model
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator for KBQA systems. Provide detailed, fair, and constructive judgments."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=AnswerJudgment,
            temperature=0.0  # Deterministic judgments
        )

        judgment = response.choices[0].message.parsed

        if judgment:
            print(f"[JUDGE] Correctness: {'✓ CORRECT' if judgment.is_correct else '✗ INCORRECT'}")
            print(f"[JUDGE] Argumentation Score: {judgment.argumentation_score}/5")
            print(f"[JUDGE] Reasoning: {judgment.correctness_reasoning[:100]}...")
            return judgment, judgment.is_correct
        else:
            print("[ERROR] Failed to parse judgment from LLM")
            return None, False

    except Exception as e:
        print(f"[ERROR] LLM judge failed: {e}")
        # Fallback to simple string matching
        simple_match = predicted_answer.lower().strip() == gold_answer.lower().strip()
        return None, simple_match


async def process_question(
    agent: KQAProAgent,
    item: Dict[str, Any],
    question_idx: int,
    total_questions: int,
    client: OpenAI,
    postprocessing_mode: str = "choice",
    sparql_wrapper: Optional[SPARQLWrapper] = None,
    judge_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single question through the agent and collect metadata.

    Args:
        agent: The KQAProAgent instance
        item: The question item from the validation set
        question_idx: Current question index (0-based)
        total_questions: Total number of questions being processed
        client: OpenAI client for answer selection/judging
        postprocessing_mode: "choice" (multiple choice), "sparql" (SPARQL synthesis), or "llm_judge" (LLM evaluation)
        sparql_wrapper: SPARQLWrapper instance (required if postprocessing_mode is "sparql")
        judge_model_name: Model name for LLM judge (required if postprocessing_mode is "llm_judge")

    Returns:
        Dictionary with results and metadata
    """
    question = item["question"]
    gold_answer = item.get("answer", None)
    gold_sparql = item.get("sparql", None)
    program = item.get("program", None)
    choices = item.get("choices", [])

    qtype = classify_question_type(question, program)

    print(f"\n{'='*80}")
    print(f"[{question_idx + 1}/{total_questions}] Processing question")
    print(f"{'='*80}")
    print(f"Q: {question}")
    print(f"Type: {qtype}")
    print(f"Gold Answer: {gold_answer}")

    start_time = time.time()

    try:
        # Process the question
        predicted_answer = await agent.ask(question)

        end_time = time.time()
        duration = end_time - start_time

        # Post-processing: Select answer based on mode
        selected_answer = None
        accuracy = False
        synthesized_sparql = None
        judgment = None

        if postprocessing_mode == "sparql":
            # SPARQL-based postprocessing
            if sparql_wrapper is None:
                raise ValueError("SPARQL wrapper is required for sparql postprocessing mode")

            selected_answer, synthesized_sparql = execute_sparql_postprocessing(
                question=question,
                agent_messages=agent._messages,
                client=client,
                sparql_wrapper=sparql_wrapper
            )

            # Calculate accuracy by comparing SPARQL result to gold answer
            if gold_answer and selected_answer:
                # For count questions, compare as numbers
                if qtype == "Count":
                    try:
                        accuracy = int(selected_answer) == int(gold_answer)
                    except (ValueError, TypeError):
                        accuracy = selected_answer.lower().strip() == gold_answer.lower().strip()
                else:
                    accuracy = selected_answer.lower().strip() == gold_answer.lower().strip()

        elif postprocessing_mode == "llm_judge":
            # LLM-based judgment postprocessing
            if judge_model_name is None:
                raise ValueError("Judge model name is required for llm_judge postprocessing mode")

            # Use the predicted answer directly
            selected_answer = predicted_answer

            # Get LLM judgment
            if gold_answer:
                judgment, accuracy = execute_llm_judge_postprocessing(
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=predicted_answer,
                    agent_messages=agent._messages,
                    client=client,
                    model_name=judge_model_name
                )
            else:
                print("[WARNING] No gold answer available for judging")

        else:
            # Choice-based postprocessing (original behavior)
            if choices and len(choices) > 0:
                selected_answer = select_answer_from_choices(
                    question=question,
                    predicted_answer=predicted_answer,
                    choices=choices,
                    client=client
                )
                # Calculate accuracy by comparing selected answer to gold answer (case-insensitive)
                if gold_answer and selected_answer:
                    accuracy = selected_answer.lower().strip() == gold_answer.lower().strip()

        # Collect metadata
        result = {
            "question": question,
            "answer": gold_answer,
            "predicted_answer": predicted_answer,
            "selected_answer": selected_answer,
            "accuracy": accuracy,
            "qtype": qtype,
            "predicted_qtype": qtype,  # Could be different if agent determines it
            "duration": f"{duration:.2f}s",
            "tokens_used": agent.token_usage.get("total_tokens", 0),
            "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.token_usage.get("completion_tokens", 0),
            "number_of_turns_used": len([m for m in agent._messages if (isinstance(m, dict) and m.get("role") == "assistant") or (hasattr(m, "role") and m.role == "assistant")]),
            "gold_sparql_query": gold_sparql,
            "extracted_sparql_query": None,  # Would need to parse from agent output
            "synthesized_sparql_query": synthesized_sparql,  # SPARQL query from postprocessing
            "postprocessing_mode": postprocessing_mode,  # Which method was used
            "program": program,
            "choices": choices,
            "success": True,
            "error": None,
            # LLM Judge fields (only populated if postprocessing_mode == "llm_judge")
            "judgment": judgment.model_dump() if judgment else None
        }

        print(f"[OK] Predicted: {predicted_answer[:100]}...")
        print(f"[OK] Selected Answer: {selected_answer} | Accuracy: {'[OK]' if accuracy else '[ERROR]'}")
        print(f"[OK] Duration: {duration:.2f}s | Tokens: {agent.token_usage.get('total_tokens', 0)}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        result = {
            "question": question,
            "answer": gold_answer,
            "predicted_answer": None,
            "selected_answer": None,
            "accuracy": False,
            "qtype": qtype,
            "predicted_qtype": None,
            "duration": f"{duration:.2f}s",
            "tokens_used": agent.token_usage.get("total_tokens", 0),
            "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.token_usage.get("completion_tokens", 0),
            "number_of_turns_used": 0,
            "gold_sparql_query": gold_sparql,
            "extracted_sparql_query": None,
            "synthesized_sparql_query": None,
            "postprocessing_mode": postprocessing_mode,
            "program": program,
            "choices": choices,
            "success": False,
            "error": str(e),
            "judgment": None
        }

        print(f"[ERROR] {e}")

    # Reset agent for next question
    await agent.reset()

    return result


def save_batch_results(
    batch_folder: Path,
    sampled_questions: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    config: Dict[str, Any]
):
    """
    Save all batch results to files.

    Args:
        batch_folder: Path to the batch folder
        sampled_questions: The sampled questions
        results: Processing results
        config: Configuration used for this batch
    """
    # Save sampled questions
    sampled_file = batch_folder / "sampled_questions.json"
    with open(sampled_file, "w", encoding="utf-8") as f:
        json.dump(sampled_questions, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved sampled questions to: {sampled_file}")

    # Save individual results
    results_file = batch_folder / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved detailed results to: {results_file}")

    # Calculate summary statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Calculate accuracy metrics
    accurate = [r for r in results if r.get("accuracy", False)]
    accuracy_rate = len(accurate) / len(results) if results else 0

    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_duration = sum(float(r["duration"].replace("s", "")) for r in results)
    avg_duration = total_duration / len(results) if results else 0

    # Count by question type
    qtype_counts = {}
    qtype_accuracy = {}
    for r in results:
        qtype = r.get("qtype", "Unknown")
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1

        # Track accuracy per question type
        if qtype not in qtype_accuracy:
            qtype_accuracy[qtype] = {"total": 0, "accurate": 0}
        qtype_accuracy[qtype]["total"] += 1
        if r.get("accuracy", False):
            qtype_accuracy[qtype]["accurate"] += 1

    # Calculate accuracy rate per question type
    qtype_accuracy_rates = {}
    for qtype, counts in qtype_accuracy.items():
        qtype_accuracy_rates[qtype] = counts["accurate"] / counts["total"] if counts["total"] > 0 else 0

    summary = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_questions": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "accurate": len(accurate),
            "accuracy_rate": accuracy_rate,
            "total_tokens_used": total_tokens,
            "total_duration_seconds": round(total_duration, 2),
            "average_duration_seconds": round(avg_duration, 2),
            "question_type_distribution": qtype_counts,
            "accuracy_by_question_type": qtype_accuracy_rates
        },
        "failed_questions": [
            {
                "question": r["question"],
                "error": r["error"]
            }
            for r in failed
        ]
    }

    # Save summary
    summary_file = batch_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved summary to: {summary_file}")

    # Save judgments if llm_judge mode was used
    if config.get("postprocessing_mode") == "llm_judge":
        judgments_with_results = []
        avg_argumentation_score = 0
        judgment_count = 0

        for r in results:
            if r.get("judgment"):
                judgment_entry = {
                    "question": r["question"],
                    "gold_answer": r["answer"],
                    "predicted_answer": r["predicted_answer"],
                    "judgment": r["judgment"],
                    "accuracy": r["accuracy"],
                    "qtype": r["qtype"]
                }
                judgments_with_results.append(judgment_entry)

                # Track average argumentation score
                if r["judgment"].get("argumentation_score"):
                    avg_argumentation_score += r["judgment"]["argumentation_score"]
                    judgment_count += 1

        avg_argumentation_score = avg_argumentation_score / judgment_count if judgment_count > 0 else 0

        judgments_file = batch_folder / "judgments.json"
        with open(judgments_file, "w", encoding="utf-8") as f:
            json.dump({
                "config": config,
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "total_judgments": len(judgments_with_results),
                    "average_argumentation_score": round(avg_argumentation_score, 2),
                    "accuracy_rate": accuracy_rate
                },
                "judgments": judgments_with_results
            }, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved LLM judgments to: {judgments_file}")
        print(f"[OK] Average Argumentation Score: {avg_argumentation_score:.2f}/5")

        # Export high-quality examples as few-shot examples
        print(f"\n[INFO] Exporting high-quality examples as few-shot examples...")
        export_counts = export_fewshot_examples_from_judgments(
            results=results,
            min_argumentation_score=4  # Only export examples with argumentation score >= 4
        )

        total_exported = sum(export_counts.values())
        if total_exported > 0:
            print(f"[OK] Exported {total_exported} new few-shot examples:")
            for qtype, count in export_counts.items():
                if count > 0:
                    print(f"  - {qtype}: {count} examples")
        else:
            print(f"[INFO] No new high-quality examples to export")

    # Print summary to console
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Postprocessing Mode: {config.get('postprocessing_mode', 'N/A')}")
    print(f"Total Questions:     {len(results)}")
    print(f"Successful:          {len(successful)}")
    print(f"Failed:              {len(failed)}")
    print(f"Success Rate:        {summary['statistics']['success_rate']:.1%}")
    print(f"Accurate:            {len(accurate)}")
    print(f"Accuracy Rate:       {accuracy_rate:.1%}")
    print(f"Total Tokens:        {total_tokens:,}")
    print(f"Total Duration:      {total_duration:.2f}s")
    print(f"Average Duration:    {avg_duration:.2f}s")
    print(f"\nQuestion Type Distribution:")
    for qtype, count in qtype_counts.items():
        acc_rate = qtype_accuracy_rates.get(qtype, 0)
        print(f"  {qtype:20s}: {count:3d} (Accuracy: {acc_rate:.1%})")

    if failed:
        print(f"\nFailed Questions ({len(failed)}):")
        for i, fail in enumerate(failed[:5], 1):  # Show first 5
            print(f"  {i}. {fail['question'][:60]}...")
            print(f"     Error: {fail['error']}")


# ============================================================================
# MAIN BATCH PROCESSING
# ============================================================================

async def run_batch(n_questions: int = 10, seed: int = 42, postprocessing_mode: str = "choice"):
    """
    Run a complete batch processing job.

    Args:
        n_questions: Number of questions to sample and process
        seed: Random seed for reproducibility
        postprocessing_mode: Postprocessing method - "choice" (multiple choice), "sparql" (SPARQL synthesis), or "llm_judge" (LLM evaluation)
    """
    print(f"\n{'='*80}")
    print("KQAPro Batch Runner")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Questions:            {n_questions}")
    print(f"  Seed:                 {seed}")
    print(f"  Postprocessing Mode:  {postprocessing_mode}")
    print(f"  Dataset:              {VALIDATION_DATASET_PATH}")
    print(f"{'='*80}\n")

    # Create batch folder
    batch_folder = get_next_batch_folder()
    print(f"[OK] Created batch folder: {batch_folder}\n")

    # Load and sample questions
    validation_data = load_validation_dataset()
    sampled_questions = sample_questions(validation_data, n_questions, seed)

    # Create agent
    agent = KQAProAgent(name="batch_runner")

    # Create OpenAI client for answer selection
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )

    # Create SPARQL wrapper if needed for sparql postprocessing mode
    sparql_wrapper = None
    if postprocessing_mode == "sparql":
        sparql_wrapper = SPARQLWrapper(VIRTUOSO_ENDPOINT)
        sparql_wrapper.setReturnFormat(JSON)
        print(f"[OK] Connected to Virtuoso endpoint: {VIRTUOSO_ENDPOINT}\n")

    # Load judge configuration if needed for llm_judge mode
    judge_model_name = None
    if postprocessing_mode == "llm_judge":
        judge_config = load_judge_config()
        judge_model_name = judge_config["model"]
        print(f"[OK] Using LLM judge: {judge_model_name}")
        print(f"[OK] Judge provider: {judge_config['provider']}\n")

    # Process all questions
    results = []
    for i, item in enumerate(sampled_questions):
        result = await process_question(
            agent=agent,
            item=item,
            question_idx=i,
            total_questions=len(sampled_questions),
            client=client,
            postprocessing_mode=postprocessing_mode,
            sparql_wrapper=sparql_wrapper,
            judge_model_name=judge_model_name
        )
        results.append(result)

    # Save results
    config = {
        "n_questions": n_questions,
        "seed": seed,
        "postprocessing_mode": postprocessing_mode,
        "dataset_path": str(VALIDATION_DATASET_PATH),
        "total_available": len(validation_data)
    }

    save_batch_results(batch_folder, sampled_questions, results, config)

    print(f"\n{'='*80}")
    print(f"[OK] Batch processing complete!")
    print(f"[OK] Results saved to: {batch_folder}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run KQAPro benchmark on a sample of validation questions"
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=10,
        help="Number of questions to sample (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--postprocessing_mode",
        type=str,
        choices=["choice", "sparql", "llm_judge"],
        default="choice",
        help="Postprocessing method: 'choice' for multiple choice selection, 'sparql' for SPARQL synthesis, 'llm_judge' for LLM-based evaluation (default: choice)"
    )

    args = parser.parse_args()

    # Run the batch
    asyncio.run(run_batch(
        n_questions=args.n_questions,
        seed=args.seed,
        postprocessing_mode=args.postprocessing_mode
    ))


if __name__ == "__main__":
    main()
