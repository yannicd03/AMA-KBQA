"""Unified postprocessing module for KBQA benchmark evaluation.

Supports multiple evaluation modes:
- choice:     Map verbose agent answers to multiple-choice options (KQAPro)
- sparql:     Synthesize & execute SPARQL from agent conversation (KQAPro)
- llm_judge:  Rich LLM judge with argumentation scoring (KQAPro & SciQA)
- simple:     String-matching fallback (SciQA)

Extracted and unified from kqapro_agent/batch_runner.py and sciqa_agent/batch_runner.py.
"""

from __future__ import annotations

import json
import os
import re
import toml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON

from ama_kbqa.config import (
    get_chat_client,
    get_chat_model_name,
    get_provider_preferences,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Color codes for terminal output
COLOR_RED = '\033[91m'
COLOR_END = '\033[0m'

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

VIRTUOSO_ENDPOINT = "http://localhost:8890/sparql"


# ============================================================================
# PYDANTIC MODEL
# ============================================================================

class AnswerJudgment(BaseModel):
    """Structured judgment of an answer's correctness and quality."""

    is_correct: bool = Field(
        description="Whether the predicted answer matches the gold answer (semantically equivalent)"
    )
    correctness_reasoning: str = Field(
        description="Detailed explanation of why the answer is correct or incorrect"
    )
    argumentation_quality: str = Field(
        description="Assessment of the agent's reasoning structure"
    )
    argumentation_score: int = Field(
        ge=1, le=5,
        description="Numeric score for argumentation quality (1=very poor to 5=excellent)"
    )
    suggested_improvement: str = Field(
        description="Specific, actionable suggestion for improvement"
    )


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class PostProcessingResult:
    """Result returned by PostProcessor.evaluate()."""
    selected_answer: Optional[str] = None
    accuracy: bool = False
    judgment: Optional[Dict] = None           # AnswerJudgment.model_dump()
    synthesized_sparql: Optional[str] = None
    intermediate_thinking: str = ""


# ============================================================================
# HELPER: Load judge configuration from config.toml
# ============================================================================

def load_judge_config() -> Dict[str, Any]:
    """Load judge configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        print("[WARNING] config.toml not found, using defaults")
        return {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v3.2",
            "temperature": 0.0,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "chat_model_provider": ""
        }

    try:
        config = toml.load(config_path)
        postprocessing = config.get("postprocessing", {})

        provider = postprocessing.get("judge_provider", "openrouter")
        provider_config = config.get(provider, {})

        judge_model = "deepseek/deepseek-v3.2"
        chat_model_provider = postprocessing.get("chat_model_provider", "")

        return {
            "provider": provider,
            "model": judge_model,
            "temperature": postprocessing.get("judge_temperature", 0.0),
            "base_url": provider_config.get("base_url", "https://openrouter.ai/api/v1"),
            "api_key_env": f"{provider.upper()}_API_KEY" if provider != "openrouter" else "OPENROUTER_API_KEY",
            "chat_model_provider": chat_model_provider
        }
    except Exception as e:
        print(f"[WARNING] Failed to load judge config: {e}, using defaults")
        return {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v3.2",
            "temperature": 0.0,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "chat_model_provider": ""
        }


# ============================================================================
# CHOICE MODE: select answer from multiple-choice options
# ============================================================================

def select_answer_from_choices(
    question: str,
    predicted_answer: str,
    choices: List[str],
    client: OpenAI
) -> str:
    """
    Use the LLM to select the most suitable answer from available choices
    based on the question and the agent's predicted answer.
    """
    valid_choices = [c for c in choices if c.lower() != "unknown"]
    if not valid_choices:
        return choices[0] if choices else "unknown"

    predicted_lower = predicted_answer.lower()

    # Exact match with emphasis markers
    for choice in valid_choices:
        if f"**{choice.lower()}**" in predicted_lower or f'"{choice.lower()}"' in predicted_lower:
            print(f"[MATCH] Found emphasized choice '{choice}' in predicted answer")
            return choice

    # Yes/no detection
    if "yes" in valid_choices and "no" in valid_choices:
        if predicted_lower.startswith("yes") or " yes," in predicted_lower or "answer: yes" in predicted_lower or "answer is yes" in predicted_lower:
            print("[MATCH] Detected 'yes' answer from predicted text")
            return "yes"
        elif predicted_lower.startswith("no") or " no," in predicted_lower or "answer: no" in predicted_lower or "answer is no" in predicted_lower:
            print("[MATCH] Detected 'no' answer from predicted text")
            return "no"

    # Exact word matches
    for choice in valid_choices:
        pattern = r'\b' + re.escape(choice.lower()) + r'\b'
        if re.search(pattern, predicted_lower):
            print(f"[MATCH] Found exact word match for choice '{choice}' in predicted answer")
            return choice

    # LLM selection
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
        call_params = {
            "model": get_chat_model_name(),
            "messages": [
                {"role": "system", "content": "You are a precise answer selector. Respond with only the number or exact text of the matching choice."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 100
        }

        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        response = client.chat.completions.create(**call_params)
        selected = response.choices[0].message.content.strip()
        print(f"[DEBUG] LLM selected: '{selected}'")

        try:
            choice_num = int(selected.split('.')[0].strip())
            if 1 <= choice_num <= len(valid_choices):
                print(f"[MATCH] LLM selected choice {choice_num}: '{valid_choices[choice_num-1]}'")
                return valid_choices[choice_num - 1]
        except (ValueError, IndexError):
            pass

        selected_lower = selected.lower()
        for choice in valid_choices:
            if choice.lower() == selected_lower:
                print(f"[MATCH] LLM selected exact match: '{choice}'")
                return choice

        for choice in valid_choices:
            if choice.lower() in selected_lower or selected_lower in choice.lower():
                print(f"[MATCH] LLM selected partial match: '{choice}'")
                return choice

        print(f"[WARNING] LLM selected '{selected}' which is not in choices. Using first choice: '{valid_choices[0]}'")
        return valid_choices[0]

    except Exception as e:
        print(f"[ERROR] Failed to select answer from choices: {e}")
        return valid_choices[0] if valid_choices else (choices[0] if choices else "unknown")


# ============================================================================
# SPARQL MODE: synthesize and execute SPARQL
# ============================================================================

def synthesize_final_sparql(
    question: str,
    agent_messages: List[Any],
    client: OpenAI
) -> Optional[str]:
    """Synthesize a single final SPARQL query from the agent's conversation history."""
    tool_interactions = []
    discovered_triples = []

    for msg in agent_messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", [])

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
            tool_interactions.append(f"Tool Result: {content[:500]}")
            try:
                if "verified_match" in content and "predicate_used" in content:
                    result_data = json.loads(content)
                    if result_data.get("verified_match"):
                        base_node = result_data.get("base_node", "")
                        predicate = result_data["verified_match"].get("predicate_used", "")
                        objects = result_data["verified_match"].get("objects", [])
                        if base_node and predicate:
                            for obj in objects[:3]:
                                obj_value = obj.get("value", "")
                                obj_type = obj.get("type", "")
                                if obj_type == "uri":
                                    discovered_triples.append(f"ex:{base_node} prop:{predicate} <{obj_value}>")
                                else:
                                    discovered_triples.append(
                                        f"ex:{base_node} prop:{predicate} ?value (found: '{obj_value}')")
            except Exception:
                pass

    context = "\n".join(tool_interactions) if tool_interactions else "No tool interactions recorded"
    if discovered_triples:
        context += "\n\nVERIFIED TRIPLE PATTERNS (use these directly):\n" + "\n".join(discovered_triples)

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

Your SPARQL query:"""

    try:
        call_params = {
            "model": get_chat_model_name(),
            "messages": [
                {"role": "system", "content": "You are a SPARQL query expert. Generate precise, executable SPARQL queries based on conversation context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 500
        }

        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        response = client.chat.completions.create(**call_params)
        query = response.choices[0].message.content.strip()

        if query.startswith("```sparql") or query.startswith("```"):
            query = query.replace("```sparql", "").replace("```", "").strip()

        lines = query.split("\n")
        query_lines = [line for line in lines if not line.strip().startswith("PREFIX")]
        query = "\n".join(query_lines).strip()

        query = query.replace("wd:", "ex:")
        query = query.replace("wdt:", "prop:")
        query = query.replace("wikibase:", "")
        query = query.replace("bd:", "")
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
    """Execute SPARQL-based postprocessing to get the final answer."""
    query = synthesize_final_sparql(question, agent_messages, client)
    if not query:
        print("[ERROR] Could not synthesize SPARQL query")
        return None, None

    full_query = f"{SPARQL_PREFIXES}\n{query}"

    try:
        sparql_wrapper.setQuery(full_query)
        results = sparql_wrapper.query().convert()

        if query.strip().upper().startswith("ASK"):
            answer = "yes" if results.get("boolean", False) else "no"
            print(f"[SPARQL] ASK query result: {answer}")
            return answer, query

        elif "COUNT" in query.upper():
            bindings = results.get("results", {}).get("bindings", [])
            if bindings and len(bindings) > 0:
                first_var = list(bindings[0].keys())[0]
                count_value = bindings[0][first_var]["value"]
                print(f"[SPARQL] COUNT query result: {count_value}")
                return str(count_value), query
            else:
                return "0", query

        else:
            bindings = results.get("results", {}).get("bindings", [])
            if bindings:
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


# ============================================================================
# INTERMEDIATE THINKING EXTRACTION
# ============================================================================

def extract_intermediate_thinking(agent_messages: List[Any]) -> str:
    """Extract intermediate thinking steps from agent message history."""
    thinking_steps = []

    for i, msg in enumerate(agent_messages):
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
                thinking_steps.append(f"\n=== Agent Reasoning (Turn {i+1}) ===")
                thinking_steps.append(content)
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get('function', {}).get('name', 'unknown')
                        tool_args = tc.get('function', {}).get('arguments', '{}')
                    else:
                        tool_name = tc.function.name
                        tool_args = tc.function.arguments
                    thinking_steps.append(f"\n=== Tool Call: {tool_name} ===")
                    thinking_steps.append(f"Arguments: {tool_args}")

        elif role == "tool" and content:
            thinking_steps.append(f"\n=== Tool Result ===")
            thinking_steps.append(content)

    return "\n".join(thinking_steps) if thinking_steps else "No intermediate thinking recorded"


# ============================================================================
# LLM JUDGE MODE (rich, with argumentation scoring)
# ============================================================================

def execute_llm_judge_postprocessing(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    agent_messages: List[Any],
    client: OpenAI,
    model_name: str,
    chat_model_provider: str = ""
) -> tuple[Optional[AnswerJudgment], bool]:
    """Use an LLM with structured output to judge answer correctness and quality."""
    reasoning_steps = []
    for msg in agent_messages:
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

    prompt = f"""You are an expert evaluator for a knowledge base question answering system.

Question: {question}

Gold (Correct) Answer: {gold_answer}

Agent's Predicted Answer: {predicted_answer}

Agent's Reasoning Process:
{reasoning_context}

Evaluate the following:

1. **Correctness**: Does the predicted answer match the gold answer? Consider semantic equivalence, not just exact string matching. For example, "yes" and "Yes, it is true" should both be considered correct if the gold answer is "yes".

2. **Correctness Reasoning**: Explain in detail why you judged the answer as correct or incorrect.

3. **Argumentation Quality**: Analyze the agent's reasoning process.

4. **Argumentation Score**: Rate the reasoning quality from 1-5:
   - 1: Very poor (major logical errors, nonsensical approach)
   - 2: Poor (significant gaps in reasoning, inefficient)
   - 3: Acceptable (reaches answer but with some issues)
   - 4: Good (solid reasoning with minor issues)
   - 5: Excellent (flawless, efficient, well-structured)

5. **Suggested Improvement**: Provide ONE specific, actionable suggestion.

You MUST respond with a valid JSON object matching this exact schema:
{{
    "is_correct": boolean,
    "correctness_reasoning": "string",
    "argumentation_quality": "string",
    "argumentation_score": integer (1-5),
    "suggested_improvement": "string"
}}

Respond ONLY with the JSON object, no additional text."""

    response = None
    try:
        call_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert evaluator for KBQA systems. You must respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 8000,
            "timeout": 60.0
        }

        if chat_model_provider:
            call_params["extra_body"] = {"provider": {"order": [chat_model_provider]}}

        response = client.chat.completions.create(**call_params)
        json_content = response.choices[0].message.content

        if json_content:
            if "```json" in json_content:
                json_content = json_content.split("```json")[1].split("```")[0].strip()
            elif "```" in json_content:
                parts = json_content.split("```")
                if len(parts) > 1:
                    json_content = parts[1].strip()

        judgment_dict = json.loads(json_content, strict=False)
        judgment = AnswerJudgment(**judgment_dict)

        if judgment:
            print(f"{COLOR_RED}[JUDGE] Correctness: {'CORRECT' if judgment.is_correct else 'INCORRECT'}{COLOR_END}")
            print(f"{COLOR_RED}[JUDGE] Argumentation Score: {judgment.argumentation_score}/5{COLOR_END}")
            print(f"{COLOR_RED}[JUDGE] Reasoning: {judgment.correctness_reasoning}{COLOR_END}")
            return judgment, judgment.is_correct
        else:
            print("[ERROR] Failed to parse judgment from LLM")
            return None, False

    except Exception as e:
        print(f"[ERROR] LLM judge failed: {e}")

        if response is not None:
            try:
                raw_content = response.choices[0].message.content if response.choices else None
                if raw_content:
                    print(f"[WARNING] Judge returned unparsed content, attempting fallback analysis")
                    content_lower = raw_content.lower()
                    is_correct = (
                        "correct" in content_lower and "incorrect" not in content_lower or
                        predicted_answer.lower().strip() == gold_answer.lower().strip()
                    )
                    print(f"[FALLBACK] Heuristic correctness: {'CORRECT' if is_correct else 'INCORRECT'}")
                    return None, is_correct
            except Exception as fallback_error:
                print(f"[WARNING] Fallback analysis also failed: {fallback_error}")

        simple_match = predicted_answer.lower().strip() == gold_answer.lower().strip()
        print(f"[FALLBACK] Using simple string matching: {'MATCH' if simple_match else 'NO MATCH'}")
        return None, simple_match


# ============================================================================
# SIMPLE MODE: string matching (SciQA fallback)
# ============================================================================

def evaluate_accuracy_simple(predicted: Optional[str], gold: str, q_type: str = "") -> bool:
    """Evaluate if predicted answer matches gold answer using string matching."""
    if predicted is None:
        return False

    pred_norm = str(predicted).lower().strip()
    gold_norm = str(gold).lower().strip()

    if pred_norm == gold_norm:
        return True

    if gold_norm in pred_norm:
        return True

    if gold_norm in ["yes", "no", "true", "false"]:
        pred_mapped = pred_norm.replace("true", "yes").replace("false", "no")
        gold_mapped = gold_norm.replace("true", "yes").replace("false", "no")
        if pred_mapped == gold_mapped:
            return True
        if pred_norm.startswith(gold_norm):
            return True

    if q_type.lower() == "count":
        try:
            pred_num = int(pred_norm.split()[0])
            gold_num = int(gold_norm)
            return pred_num == gold_num
        except (ValueError, IndexError):
            pass

    return False


# ============================================================================
# POSTPROCESSOR CLASS
# ============================================================================

class PostProcessor:
    """Unified postprocessor that dispatches to the appropriate evaluation mode."""

    def __init__(self, mode: str, agent_name: str):
        """
        Args:
            mode: Evaluation mode (choice, sparql, llm_judge, simple)
            agent_name: Agent name (kqapro, sciqa)
        """
        self.mode = mode
        self.agent_name = agent_name

        # Validate mode vs agent
        valid_modes = {
            "kqapro": ["choice", "sparql", "llm_judge"],
            "sciqa": ["llm_judge", "simple"],
        }
        allowed = valid_modes.get(agent_name, ["llm_judge", "simple"])
        if mode not in allowed:
            raise ValueError(
                f"Postprocessing mode '{mode}' is not valid for agent '{agent_name}'. "
                f"Allowed modes: {allowed}"
            )

        # Initialize mode-specific resources
        self._client: Optional[OpenAI] = None
        self._judge_client: Optional[OpenAI] = None
        self._judge_model_name: Optional[str] = None
        self._judge_chat_model_provider: str = ""
        self._sparql_wrapper: Optional[SPARQLWrapper] = None

        self._init_resources()

    def _init_resources(self):
        """Initialize resources needed for the configured mode."""
        if self.mode in ("choice", "sparql"):
            self._client = get_chat_client()

        if self.mode == "sparql":
            self._sparql_wrapper = SPARQLWrapper(VIRTUOSO_ENDPOINT)
            self._sparql_wrapper.setReturnFormat(SPARQL_JSON)
            print(f"[OK] Connected to Virtuoso endpoint: {VIRTUOSO_ENDPOINT}")

        if self.mode == "llm_judge":
            judge_config = load_judge_config()
            self._judge_model_name = judge_config["model"]
            self._judge_chat_model_provider = judge_config.get("chat_model_provider", "")

            judge_api_key = os.getenv(judge_config["api_key_env"])
            if not judge_api_key:
                raise RuntimeError(f"{judge_config['api_key_env']} missing in .env for judge")

            self._judge_client = OpenAI(
                base_url=judge_config["base_url"],
                api_key=judge_api_key
            )
            print(f"[OK] Using LLM judge: {self._judge_model_name}")
            print(f"[OK] Judge provider: {judge_config['provider']}")

    def evaluate(
        self,
        question: str,
        predicted: Optional[str],
        gold: str,
        agent_messages: List[Any],
        choices: Optional[List[str]] = None,
        program: Optional[List] = None,
        q_type: str = "Unknown",
    ) -> PostProcessingResult:
        """
        Evaluate a single answer using the configured postprocessing mode.

        Args:
            question: Original question text
            predicted: Agent's predicted answer
            gold: Gold/correct answer
            agent_messages: Agent's full message history
            choices: Multiple-choice options (choice mode only)
            program: KQAPro program structure (optional)
            q_type: Question type string

        Returns:
            PostProcessingResult with evaluation details
        """
        result = PostProcessingResult()
        result.intermediate_thinking = extract_intermediate_thinking(agent_messages)

        if self.mode == "choice":
            result = self._evaluate_choice(question, predicted, gold, choices, agent_messages)
        elif self.mode == "sparql":
            result = self._evaluate_sparql(question, predicted, gold, agent_messages, q_type)
        elif self.mode == "llm_judge":
            result = self._evaluate_llm_judge(question, predicted, gold, agent_messages)
        elif self.mode == "simple":
            result = self._evaluate_simple(predicted, gold, q_type)

        # Always attach intermediate thinking
        result.intermediate_thinking = extract_intermediate_thinking(agent_messages)
        return result

    def _evaluate_choice(
        self, question, predicted, gold, choices, agent_messages
    ) -> PostProcessingResult:
        result = PostProcessingResult()
        if choices and len(choices) > 0 and predicted:
            result.selected_answer = select_answer_from_choices(
                question=question,
                predicted_answer=predicted,
                choices=choices,
                client=self._client
            )
            if gold and result.selected_answer:
                result.accuracy = result.selected_answer.lower().strip() == gold.lower().strip()
        return result

    def _evaluate_sparql(
        self, question, predicted, gold, agent_messages, q_type
    ) -> PostProcessingResult:
        result = PostProcessingResult()
        selected_answer, synthesized_sparql = execute_sparql_postprocessing(
            question=question,
            agent_messages=agent_messages,
            client=self._client,
            sparql_wrapper=self._sparql_wrapper
        )
        result.selected_answer = selected_answer
        result.synthesized_sparql = synthesized_sparql

        if gold and selected_answer:
            if q_type == "Count":
                try:
                    result.accuracy = int(selected_answer) == int(gold)
                except (ValueError, TypeError):
                    result.accuracy = selected_answer.lower().strip() == gold.lower().strip()
            else:
                result.accuracy = selected_answer.lower().strip() == gold.lower().strip()
        return result

    def _evaluate_llm_judge(
        self, question, predicted, gold, agent_messages
    ) -> PostProcessingResult:
        result = PostProcessingResult()
        result.selected_answer = predicted

        if gold and predicted:
            judgment, accuracy = execute_llm_judge_postprocessing(
                question=question,
                gold_answer=gold,
                predicted_answer=predicted,
                agent_messages=agent_messages,
                client=self._judge_client,
                model_name=self._judge_model_name,
                chat_model_provider=self._judge_chat_model_provider
            )
            result.accuracy = accuracy
            result.judgment = judgment.model_dump() if judgment else None
        else:
            print("[WARNING] No gold answer available for judging")
        return result

    def _evaluate_simple(self, predicted, gold, q_type) -> PostProcessingResult:
        result = PostProcessingResult()
        result.selected_answer = predicted
        result.accuracy = evaluate_accuracy_simple(predicted, gold, q_type)
        return result
