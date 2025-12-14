from __future__ import annotations
from ama_kbqa.config import (
    get_chat_client,
    get_chat_model_name,
    get_chat_temperature,
    get_chat_max_tokens,
    get_provider_preferences,
)
import os
import sys
import asyncio
import json
import inspect
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import AsyncExitStack
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError

# Configure stdout to handle Unicode on Windows
if sys.platform == 'win32':
    import codecs
    # Reconfigure stdout to use UTF-8 with error handling
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    else:
        # Fallback for older Python versions
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv(override=True)

# Import configuration utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- TRACING & COLORS ---

COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'  # Added Cyan for better distinction
COLOR_END = '\033[0m'


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardized tracing with timestamp and agent prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Handle multi-line messages nicely by indenting subsequent lines
    lines = msg.split('\n')
    header = f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {lines[0]}"
    print(header)
    for line in lines[1:]:
        print(f"{' ' * (len(timestamp) + 2 + len(agent_name) + 6)} {color}{line}{COLOR_END}")


# Dynamic path resolution for the Sub-Agent Server
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_subagent_server_path = ama_kbqa_root / "server" / "kqapro_server.py"

# --- CONFIGURATION ---
# Configuration is now loaded from config.toml via ama_kbqa.config
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MCP_SERVER_PATH = os.getenv("SUBAGENT_SERVER_PATH", str(default_subagent_server_path))


# --- MCP CLIENT FOR SUB-AGENTS (Adapted) ---

class MCPClient:
    def __init__(self, server_path: str, agent_name: str):
        self.server_path = Path(server_path)
        self.agent_name = agent_name
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False

    async def start(self):
        if self._connected:
            return
        if not self.server_path.exists():
            trace(self.agent_name, f"{COLOR_RED}MCP server not found: {self.server_path}{COLOR_END}", COLOR_RED)
            raise FileNotFoundError(f"MCP server not found: {self.server_path}")

        client_gen = stdio_client(StdioServerParameters(command=sys.executable, args=[str(self.server_path)], env=None))

        read, write = await self.exit_stack.enter_async_context(client_gen)

        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self._connected = True

    async def list_tools(self) -> List[McpTool]:
        if not self.session:
            raise RuntimeError("Not connected")
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, args: Dict) -> str:
        if not self.session:
            raise RuntimeError("Not connected")
        result = await self.session.call_tool(name, arguments=args)
        if hasattr(result, "content") and result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        if self._connected:
            try:
                await self.exit_stack.aclose()
                # Increased delay to allow subprocess cleanup and stdio buffer flushing
                # This prevents zombie processes and termination hangs
                await asyncio.sleep(0.5)  # Increased from 0.05s to 0.5s
            except (CancelledError, RuntimeError) as e:
                trace(
                    self.agent_name, f"{COLOR_YELLOW}WARNING: MCP-Close error on shutdown ignored ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False
                # Additional cleanup delay to ensure resources are fully released
                await asyncio.sleep(0.2)


# --- KQAProAgent ---

class KQAProAgent:
    """
    KQAPro Agent with deterministic pre/post processing hooks.

    The agent uses qtype-specific reasoning strategies that are injected
    dynamically based on the question classification.
    """

    # Question Type Specific Reasoning Strategies
    # These are injected during the pre-agent hook based on QtypePrediction results
    QTYPE_STRATEGIES = {
        "Count": """
        STRATEGY: Count (Aggregation)
        Topology: [Entity] -> [Predicate] -> [Target_Nodes]
        Strategy:
            1. **Small Sets (Estimate < 20 items):** Use `GetRelationDetails` to fetch the connected nodes, then count the length of the list in your final answer.
            2. **Large/Global Sets (Unknown size):** DO NOT attempt to fetch all items (e.g., "How many cities in China?"). You must use `RunSPARQL` to perform the count database-side.
        SPARQL Fallback:
        SELECT (COUNT(DISTINCT ?target) AS ?count) WHERE {
        ex:ENTITY_ID prop:PREDICATE ?target .
        # Optional: ?target prop:instance_of ex:TARGET_TYPE .
        }
        """,

        "QueryAttr": """
        STRATEGY: QueryAttr (Direct Lookup)
        Topology: [Entity] -> [Predicate] -> [Target]
        Strategy:
            1. **Standard Lookup:** If you have the Entity ID and need an attribute, use `GetAttributeDetails` (for literals) or `GetRelationDetails` (for linked entities).
            2. **Reverse Lookup:** If you have a unique value (e.g., "UKE11", "ISBN-13", "http://...") and need the Entity, use `FindByAttribute`.
        SPARQL Fallback:
        SELECT ?value WHERE {
        ex:ENTITY_ID prop:PREDICATE ?value .
        }
        """,

        "QueryAttrQualifier": """
        STRATEGY: QueryAttrQualifier (Contextual Fact)
        Topology: [Entity] <-[is_subject_of]- [Fact_Node] -[has_qualifier]-> [Value]
        Strategy:
            1. **Primary Tool:** Use `GetEdgeQualifiers`. This tool is specifically designed to fetch metadata (time, location, role) attached to a specific relationship.
            2. **Pre-requisite:** You must first find the target value of the main relation (via `GetRelationDetails`) before you can query its qualifiers.
        SPARQL Fallback:
        SELECT ?qualifier_value WHERE {
        # Reified statement pattern
        ?fact_node pred:fact_h ex:ENTITY_ID ;
                    pred:fact_r prop:PREDICATE ;
                    pred:fact_t "TARGET_VALUE" . # or <TARGET_URI>
        ?fact_node qual:QUALIFIER_PREDICATE ?qualifier_value .
        }
        """,

        "QueryName": """
        STRATEGY: QueryName (Reverse Lookup / Identification)
        Topology: [Target?] -> [Predicate] -> [Known_Object]
        Strategy:
            1. **Unique ID/Code:** If the question contains a unique identifier (ID, code, URL), use `FindByAttribute` immediately.
            2. **Multi-Condition/Intersection:** If the question implies a complex filter (e.g., "Actors born in Boston who played Batman"), DO NOT fetch lists and intersect them manually. Use `RunSPARQL` to handle the intersection efficiently.
        SPARQL Fallback:
        SELECT ?subjectLabel WHERE {
        ?subject prop:PREDICATE_1 ex:OBJECT_ID_1 .
        ?subject prop:PREDICATE_2 ex:OBJECT_ID_2 .
        ?subject rdfs:label ?subjectLabel .
        }
        """,

        "QueryRelation": """
        STRATEGY: QueryRelation (Relationship Identification)
        Topology: [Entity_A] <-> [?Predicate] <-> [Entity_B]
        Strategy:
            1. **Standard:** Use `GetRelationDetails` on Entity A to see if Entity B appears in the results.
            2. **Bidirectional Check:** If A->B fails, check B->A. The relation might be defined inversely in the graph.
        SPARQL Fallback:
        SELECT DISTINCT ?p ?label WHERE {
        { ex:ENTITY_A_ID ?p ex:ENTITY_B_ID . }
        UNION
        { ex:ENTITY_B_ID ?p ex:ENTITY_A_ID . }
        ?p rdfs:label ?label .
        }
        """,

        "QueryRelationQualifier": """
        STRATEGY: QueryRelationQualifier (Relation Detail)
        Topology: [Entity_A] <-[in_statement]- [Fact_Node] -[has_role]-> [Role_Value]
        Strategy:
            1. **Identify Connection:** Use `GetRelationDetails` to confirm A and B are connected.
            2. **Extract Detail:** Use `GetEdgeQualifiers` on that specific connection to find the role, capacity, or nuance requested.
        SPARQL Fallback:
        SELECT ?detail_value WHERE {
        ?fact_node pred:fact_h ex:ENTITY_A_ID ;
                    pred:fact_r prop:PREDICATE ;
                    pred:fact_t ex:ENTITY_B_ID .
        ?fact_node qual:DETAIL_PREDICATE ?detail_value .
        }
        """,

        "SelectAmong": """
        STRATEGY: SelectAmong (Superlative/Sorting)
        Topology: [Group] -> [Member] -> [Value]
        Strategy:
            1. **Closed Group (< 20 items):** If the group is small (e.g., "Who is the tallest of these 3 brothers?"), use `CompareEntities` to sort them.
            2. **Open/Global Group:** If the group is large (e.g., "Longest movie ever"), DO NOT fetch all items. Use `RunSPARQL` with `ORDER BY` and `LIMIT 1`.
        SPARQL Fallback:
        SELECT ?itemLabel ?value WHERE {
        ?item prop:instance_of ex:GROUP_ID .
        ?item attr:SORT_ATTRIBUTE ?value .
        ?item rdfs:label ?itemLabel .
        }
        ORDER BY DESC(?value) # Use ASC(?value) for 'Smallest'/'First'
        LIMIT 1
        """,

        "SelectBetween": """
        STRATEGY: SelectBetween (Binary Comparison)
        Topology: [Entity_A/B] -> [Attribute] -> [Value]
        Strategy:
            1. **Primary Tool:** ALWAYS use `CompareEntities`. This handles fetching and sorting in a single step.
            2. **Avoid:** Do not use `GetAttributeDetails` individually for each entity, as this increases cognitive load and risks "forgetting" to compare.
        SPARQL Fallback:
        SELECT ?itemLabel ?value WHERE {
        VALUES ?item { ex:ENTITY_A_ID ex:ENTITY_B_ID }
        ?item attr:ATTRIBUTE ?value .
        ?item rdfs:label ?itemLabel .
        }
        ORDER BY DESC(?value)
        LIMIT 1
        """,

        "Verify": """
        STRATEGY: Verify (Boolean Check)
        Topology: [Entity] -> [Attribute] ? [Value]
        Strategy:
            1. **Primary Tool:** Retrieve the attribute value using `GetAttributeDetails`, then IMMEDIATELY pass it to `VerifyNumericCondition` to get a True/False verdict.
            2. **Logic:** Do not rely on your internal training to compare numbers (e.g. 146 vs 238.9). Use the tool.
        SPARQL Fallback:
        ASK {
        ex:ENTITY_ID attr:ATTRIBUTE ?value .
        FILTER (?value > "TARGET_VALUE"^^xsd:decimal)
        }
        """,

        "Query": """
        STRATEGY: General Query
        Strategy:
            1. Use FindNode to locate relevant entities.
            2. Use `GetAttributeDetails` and `GetRelationDetails` to gather information.
            3. Use `ExploreNeighborhood` if you need to find related entities via semantic search.
            4. Use `RunSPARQL` as a fallback for complex queries that don't fit the patterns above.
            5. Accumulate findings in your journal and review before answering.
        """
    }

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        # Get chat client and model from config.toml
        try:
            self.client = get_chat_client()
            self.model = get_chat_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(
                f"Failed to initialize LLM client from config.toml: {e}"
            )

        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        # NEW: Token Tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # Stuck Loop Detection: Track recent tool calls to prevent infinite loops
        self.tool_call_history = []  # List of (tool_name, args_json_str) tuples

        self.system_prompt = """SYSTEM ROLE
            You are the KQAPro Execution Agent. Your goal is to answer natural language questions by querying a Knowledge Graph (KG).

            CRITICAL RULES
            1.  **No Hallucination:** You have NO internal knowledge. You MUST verify every fact using the tools. NEVER answer without using tools.
            2.  **Schema Compliance:** You must use the valid predicates returned by tools. Do not guess predicate names (e.g., do not guess `wdt:P123`, find it first).
            3.  **State Management:** Use `ManageJournal` to track progress and avoid loops.
            4.  **Pivot Logic (Dead End Detection):** If a specific search strategy fails twice (e.g., searching for "Barbara McLean" yields 0 results), you MUST PIVOT. Do not try the same term a third time.
                * *Pivot Strategy:* Switch to searching for the *connected* entity (e.g., search for the Award name instead of the Person) and filter down.
            5.  **Complete Retrieval:** After FindNode returns available attributes/predicates, you MUST call GetAttributeDetails or GetRelationDetails to get actual values.

            KNOWLEDGE GRAPH SPECIFICS (CRITICAL)
            You are operating on a specific ontology. You MUST use the following prefixes in your thought process and SPARQL construction. DO NOT define these in your `RunSPARQL` calls; the server injects them automatically.

            * `ex:` -> Entities (e.g., `ex:Q64`)
            * `prop:` -> Properties/Relations (e.g., `prop:P1082`)
            * `attr:` -> Attributes
            * `qual:` -> Qualifiers
            * `unit:` -> Units

            YOUR WORKING MEMORY (SCRATCHPAD) - AUTO-UPDATED!
            The system automatically maintains a scratchpad (journal) that tracks:
             visited_nodes - Auto-updated when you call FindNode
             found_values - Auto-updated when you call GetAttributeDetails
             verified_facts - Auto-updated by tools
             completed_steps - Auto-updated to track progress
             failed_attempts - Auto-logged when tools fail

            **MANDATORY BEFORE ANSWERING:**
            Call GetJournalSummary() before giving your final answer!
            This shows ALL values you discovered. Your answer MUST be based on these values.
            If a value isn't in the journal summary, you haven't found it yet!

            KNOWLEDGE GRAPH ACCESS TOOLS (TWO-TIER PATTERN)
            The system uses an efficient two-tier data access pattern to minimize context usage:

            TIER 1 - DISCOVERY (Lightweight Schema Exploration):
            • FindNode(semantic_node_name): Performs semantic vector search to find relevant entities/concepts.
              Use this for natural language concepts (e.g., "Boston", "Director").

            • FindByAttribute(value, attribute_name): Performs precise reverse lookup for entities by value.
              Args: value (e.g. "UKE11", "http://...", "94332"), attribute_name (e.g. "NUTS code", "official website")
              * *Constraint:* If the user query contains a **Unique ID**, **URL**, or **Technical Code**, you MUST use this tool instead of FindNode. It is faster and exact.

            TIER 2 - RETRIEVAL (Targeted Value Fetching):
            Once you know what's available from FindNode, use these tools to get specific values:

            • GetAttributeDetails(base_node_id, attribute_name): Queries Virtuoso for full attribute details.
              Use this when you need the actual value of an attribute discovered via FindNode.

            • GetRelationDetails(base_node_id, relation_name): Queries Virtuoso for nodes connected via a relation.
              Use this when you need to find what entities are connected via a specific relation.

            • ExploreNeighborhood(base_node_id, semantic_relation_name): Semantic search + SPARQL verification.
              Use this when you don't know the exact predicate name and need to find it semantically.

            TIER 3 - QUALIFIERS & METADATA (Contextual Data):
            **CRITICAL FOR ACCURACY:** If the question asks for "when", "where", "language of", or "role" concerning a specific link (e.g., "language of the website", "publication date in Germany"):
            • GetEdgeQualifiers(subject_id, predicate_name, target_id): Retrieves 'facts about a fact'.
              * *Rule:* If you find a relation (e.g., Movie -> website -> URL) and need details *about* that link (like its language), you MUST use this tool. The language is likely a qualifier on the website edge, not a global attribute of the movie.

            TIER 4 - VERIFICATION (The Math Judge):
            • VerifyNumericCondition(value1, operator, value2):
              * *Rule:* NEVER perform mental math comparison. If the question asks "Is X > Y?", retrieve X and Y, then pass them to this tool. Trust the tool's True/False verdict over your own generation.

            EXECUTION LOOP (General Strategy)
            1. **Analyze Strategy:** Read the pre-analysis provided in the chat history.
            2. **Identify Entry Point:**
               * If Unique ID present -> `FindByAttribute`.
               * If Named Entity -> `FindNode`.
            3. **Retrieve Values:** Fetch actual data using Tier 2 tools.
            4. **Check Qualifiers:** If the question implies context (time, place, language of a specific thing), use `GetEdgeQualifiers`.
            5. **Pivot if Stuck:** If a search returns 0 results twice, STOP searching that term. Try a neighbor or a broader category.
            6. **Verify:** Use `VerifyNumericCondition` for any numbers/dates.
            7. **Synthesize:** Call `GetJournalSummary` and formulate your answer.
            """

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _trace(self, msg: str, color: str = COLOR_GREEN):
        trace(self.name, msg, color)

    def _detect_stuck_loop(self, func_name: str, func_args: dict, threshold: int = 3) -> bool:
        """
        Detect if the agent is stuck calling the same tool with same parameters.

        Args:
            func_name: Name of the tool being called
            func_args: Arguments for the tool call
            threshold: Number of identical calls to trigger detection (default: 3)

        Returns:
            True if stuck loop detected, False otherwise
        """
        # Create a hashable representation of this tool call
        # Sort keys to ensure consistent comparison
        args_str = json.dumps(func_args, sort_keys=True)
        current_call = (func_name, args_str)

        # Add to history
        self.tool_call_history.append(current_call)

        # Keep only recent history (last 10 calls to avoid memory bloat)
        if len(self.tool_call_history) > 10:
            self.tool_call_history.pop(0)

        # Check if the last N calls are identical
        if len(self.tool_call_history) >= threshold:
            recent_calls = self.tool_call_history[-threshold:]
            if all(call == current_call for call in recent_calls):
                return True

        return False

    def _mcp_tool_to_openai(self, mcp_tool: McpTool) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }

    async def _init_mcp(self):
        if self.mcp:
            return
        try:
            self._trace(f"Starting own MCP server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("Own MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}Own MCP error: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None
            raise

    async def ask(self, query: str) -> str:
        self._trace(f"Incoming query: '{query}'", COLOR_GREEN)

        try:
            # 1. Start MCP and load tools
            await self._init_mcp()

            if not self.mcp:
                self._trace(f"{COLOR_YELLOW}Tool server not available. Answering without tools.{COLOR_END}", COLOR_YELLOW)
                self._messages.append({"role": "user", "content": query})
                return self._llm_call_text_only()

            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
            self._trace(f"Found {len(openai_tools)} tools.")

            # 2. Add user query to message history
            self._messages.append({"role": "user", "content": query})

            # ═══════════════════════════════════════════════════════════════════════
            # PRE-AGENT HOOK: Deterministic Classification & Entity Extraction
            # This ALWAYS runs before the main loop to provide foundational analysis
            # ═══════════════════════════════════════════════════════════════════════
            self._trace("🎯 Starting pre-agent classification hook", COLOR_CYAN)

            # Step 1: Run QtypePrediction and EntityExtraction
            self._trace("🔍 Running QtypePrediction and EntityExtraction...", COLOR_YELLOW)

            # Call both tools (they're independent, could be parallelized in future)
            # For now, call sequentially for clear error handling and logging
            try:
                qtype_result = await self.mcp.call_tool("QtypePrediction", {"question": query})
                self._trace(f"✓ QtypePrediction complete", COLOR_GREEN)
            except Exception as e:
                self._trace(f"⚠️ QtypePrediction failed: {e}", COLOR_YELLOW)
                qtype_result = json.dumps({"qtype": "Query", "fewshot_examples": ""})

            try:
                entities_result = await self.mcp.call_tool("EntityExtraction", {"question": query})
                self._trace(f"✓ EntityExtraction complete", COLOR_GREEN)
            except Exception as e:
                self._trace(f"⚠️ EntityExtraction failed: {e}", COLOR_YELLOW)
                entities_result = json.dumps({"entities": [], "relations": []})

            # Step 2: Parse results (tools return JSON strings)
            try:
                qtype_data = json.loads(qtype_result) if isinstance(qtype_result, str) else qtype_result
                entities_data = json.loads(entities_result) if isinstance(entities_result, str) else entities_result
            except json.JSONDecodeError as e:
                self._trace(f"⚠️ Failed to parse pre-hook results: {e}", COLOR_YELLOW)
                qtype_data = {"qtype": "Query", "fewshot_examples": ""}
                entities_data = {"entities": [], "relations": []}

            # Step 3: Extract and format the analysis data
            qtype = qtype_data.get("qtype", "Query")
            fewshot_examples = qtype_data.get("fewshot_examples", "")
            entities = entities_data.get("entities", [])
            relations = entities_data.get("relations", [])

            # Format entities and relations for readability
            formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none identified)"
            formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none identified)"

            # Step 4: Get the qtype-specific strategy
            qtype_strategy = self.QTYPE_STRATEGIES.get(qtype, self.QTYPE_STRATEGIES["Query"])

            # Step 5: Build the analysis context message
            analysis_context = f"""═══════════════════════════════════════════════════════════════════════
PRE-ANALYSIS (Automatically Computed)
═══════════════════════════════════════════════════════════════════════

Question Type: {qtype}

Extracted Entities:
{formatted_entities}

Extracted Relations:
{formatted_relations}

{qtype_strategy}
"""

            # Add fewshot examples if provided by QtypePrediction
            if fewshot_examples and fewshot_examples.strip():
                analysis_context += f"""
Relevant Few-Shot Examples for {qtype} Questions:
{fewshot_examples}
"""

            analysis_context += """
═══════════════════════════════════════════════════════════════════════

Use this pre-analysis to guide your tool selection and reasoning strategy.
The strategy above is specifically tailored for this question type.
Now proceed with your investigation using the available tools."""

            # Step 6: Inject the analysis context as a user message
            self._messages.append({
                "role": "user",
                "content": analysis_context
            })

            self._trace(
                f"✅ Pre-agent hook complete - Type: {qtype}, Entities: {len(entities)}, Relations: {len(relations)}", COLOR_GREEN)

            # 3. Iterative tool-call loop
            iteration_count = 0
            max_iterations = 50  # Safety limit to prevent infinite loops
            while True:
                iteration_count += 1
                self._trace(f"🔄 Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

                # Safety check: prevent infinite loops
                if iteration_count > max_iterations:
                    self._trace(
                        f"⚠️  WARNING: Reached maximum iteration limit ({max_iterations}). Stopping.", COLOR_RED)
                    self._trace(f"⚠️  The agent may not have found a complete answer. Review logs for issues.", COLOR_RED)
                    return "Error: Agent reached maximum iteration limit without completing the task. This may indicate a problem with the reasoning loop or insufficient information in the knowledge base."

                # Warn when approaching iteration limit
                if iteration_count >= max_iterations - 5:
                    self._trace(f"⚠️  Approaching iteration limit! ({iteration_count}/{max_iterations})", COLOR_YELLOW)

                # Periodic status update
                if iteration_count % 10 == 0:
                    self._trace(
                        f"📊 Status: {self.token_usage['total_tokens']} tokens used, {len(self._messages)} messages in history", COLOR_CYAN)

                # Periodic journal refresh (every 5 iterations)
                if iteration_count % 5 == 0 and iteration_count > 0:
                    self._trace("🔄 Injecting journal summary for working memory refresh", COLOR_CYAN)
                    try:
                        journal_refresh = await self.mcp.call_tool("GetJournalSummary", {})
                        self._messages.append({
                            "role": "user",
                            "content": f"""📋 WORKING MEMORY REFRESH (Iteration {iteration_count})

Here's everything you've discovered so far:

{journal_refresh}

Continue your investigation using this information. Avoid revisiting nodes or queries you've already completed."""
                        })
                    except Exception as e:
                        self._trace(f"⚠️ Failed to inject journal refresh: {e}", COLOR_YELLOW)

                self._trace(
                    f"📤 Calling LLM with {len(self._messages)} messages and {len(openai_tools)} tools...", COLOR_YELLOW)
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message

                finish_reason = response.choices[0].finish_reason
                self._trace(f"📥 LLM response received (finish_reason: {finish_reason})", COLOR_CYAN)

                # Warn on unexpected finish reasons
                if finish_reason not in ["stop", "tool_calls"]:
                    self._trace(
                        f"⚠️  Unexpected finish_reason: {finish_reason} (expected 'stop' or 'tool_calls')", COLOR_YELLOW)
                    if finish_reason == "length":
                        self._trace(
                            f"⚠️  LLM hit token limit! Response may be truncated. Consider increasing max_tokens.", COLOR_RED)

                # Update Token Usage
                if response.usage:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens

                # Logging intermediate thoughts (Thoughts/Text)
                if message.content:
                    self._trace(f"🧠 Thought/Text: {message.content}", COLOR_BLUE)

                # Final answer (no more tools) - break out of loop for synthesis
                if not message.tool_calls:
                    self._trace(f"🏁 No more tool calls - breaking to synthesis step", COLOR_GREEN)
                    # Store any intermediate reasoning the agent provided
                    if message.content:
                        self._messages.append({"role": "assistant", "content": message.content})
                    # Break out of loop to run deterministic synthesis
                    break

                # --- CORRECTION START ---
                # The assistant message MUST be added to the history ONCE
                # BEFORE processing the tools.
                # Convert ChatCompletionMessage to dict for message history
                msg_dict = {
                    "role": message.role,
                    "content": message.content,
                }
                if message.tool_calls:
                    # Convert tool_calls to proper format
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                self._messages.append(msg_dict)
                # --- CORRECTION END ---

                # Track if GetJournalSummary was called in this turn
                called_get_journal_summary = False

                # Execute tool call(s)
                self._trace(f"🔧 Processing {len(message.tool_calls)} tool call(s)", COLOR_YELLOW)
                for tool_call in message.tool_calls:
                    # Validate tool call has a valid name
                    if not tool_call.function or not tool_call.function.name:
                        self._trace(f"⚠️  Skipping invalid tool call with missing name", COLOR_RED)
                        # Add error response to maintain conversation flow
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "invalid_tool",
                            "content": "Error: Tool call has no valid name. Please try a different approach."
                        })
                        continue

                    func_name = tool_call.function.name

                    # Track if this is GetJournalSummary
                    if func_name == "GetJournalSummary":
                        called_get_journal_summary = True

                    try:
                        # Handle case where arguments might be None
                        args_str = tool_call.function.arguments
                        if args_str is None or args_str == "":
                            func_args = {}
                        else:
                            func_args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        self._trace(f"⚠️  Invalid JSON in tool arguments: {e}", COLOR_YELLOW)
                        func_args = {}

                    # Logging
                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"🛠️  Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    # STUCK LOOP DETECTION: Check if this tool+params has been called 3 times
                    if self._detect_stuck_loop(func_name, func_args):
                        self._trace(
                            f"🔄 STUCK LOOP DETECTED: {func_name} called 3 times with identical parameters!",
                            COLOR_RED
                        )

                        # Fetch current journal state to help agent recover
                        try:
                            journal_state = await self.mcp.call_tool("GetJournalSummary", {})
                            self._trace("📋 Fetched journal summary for stuck loop recovery", COLOR_CYAN)
                        except Exception as e:
                            self._trace(f"⚠️ Failed to fetch journal for stuck loop: {e}", COLOR_YELLOW)
                            journal_state = "(Journal unavailable)"

                        # Don't execute the tool - instead provide intervention message with journal
                        tool_result = (
                            f"⚠️ **STUCK LOOP DETECTED**\n\n"
                            f"You have called `{func_name}` with the same parameters 3 times in a row. "
                            f"This suggests your current approach is not working.\n\n"
                            f"**Please try a different strategy:**\n"
                            f"- Use a different tool to gather information\n"
                            f"- Modify your search parameters (try synonyms, broader/narrower terms)\n"
                            f"- Change your reasoning approach\n"
                            f"- Use RunSPARQL as a fallback for direct queries\n\n"
                            f"**Repeated parameters were:**\n{args_pretty}\n\n"
                            f"═══════════════════════════════════════════════════════════════════════\n"
                            f"📋 HERE'S WHAT YOU'VE ALREADY DISCOVERED:\n"
                            f"═══════════════════════════════════════════════════════════════════════\n\n"
                            f"{journal_state}\n\n"
                            f"Use this information to try a completely different approach."
                        )

                        # Clear history for this specific call to allow different attempts
                        args_str = json.dumps(func_args, sort_keys=True)
                        self.tool_call_history = [
                            call for call in self.tool_call_history
                            if call != (func_name, args_str)
                        ]
                    else:
                        # Normal execution - no loop detected
                        # Execution with error handling
                        try:
                            tool_result = await self.mcp.call_tool(func_name, func_args)

                            # Logging result
                            log_result = tool_result
                            if len(log_result) > 500:
                                log_result = log_result[:500] + f"... [truncated, total len: {len(tool_result)}]"
                            self._trace(f"🔙 Result ({func_name}): {log_result}", COLOR_CYAN)

                        except Exception as tool_error:
                            # Handle tool execution errors gracefully
                            error_msg = str(tool_error)
                            self._trace(f"⚠️  Tool {func_name} failed: {error_msg}", COLOR_RED)

                            # Create error response that helps the LLM recover
                            tool_result = f"Error executing {func_name}: {error_msg}. Please try a different approach or verify the parameters."

                    # --- BUG WAS HERE (Removed: self._messages.append(message)) ---

                    # Only the tool result is appended in the loop
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

                # CRITICAL FIX: After ALL tools are processed, if GetJournalSummary was called,
                # inject a user prompt to force the agent to provide a final answer
                if called_get_journal_summary:
                    self._trace("📝 GetJournalSummary was called - injecting answer prompt", COLOR_CYAN)
                    self._messages.append({
                        "role": "user",
                        "content": "You have reviewed everything you discovered in your journal. Now you MUST provide your final answer to the original question as clear, direct text. Do NOT call any more tools."
                    })

            # ═══════════════════════════════════════════════════════════════════════
            # DETERMINISTIC POST-AGENT SYNTHESIS STEP
            # This ALWAYS runs after tool gathering is complete to ensure a final answer
            # ═══════════════════════════════════════════════════════════════════════
            self._trace("🎯 Starting deterministic synthesis step", COLOR_CYAN)

            # Step 1: Fetch complete journal summary with all discovered information
            self._trace("📋 Fetching journal summary with all discovered data...", COLOR_YELLOW)
            journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
            self._trace(f"✓ Journal fetched ({len(journal_summary)} chars)", COLOR_GREEN)

            # Step 2: Inject synthesis prompt with full context
            synthesis_prompt = f"""You have completed your tool-based investigation. Here is EVERYTHING you discovered during your research:

═══════════════════════════════════════════════════════════════════════
JOURNAL SUMMARY - ALL DISCOVERED INFORMATION
═══════════════════════════════════════════════════════════════════════

{journal_summary}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════

Based ONLY on the information shown above in your journal, provide a clear, direct, and complete answer to this question:

"{query}"

INSTRUCTIONS:
- Use ONLY the facts and values from your journal summary above
- Provide a direct answer without explaining your process
- If the information is insufficient to answer completely, state exactly what is missing
- Be concise but complete

YOUR FINAL ANSWER:"""

            self._messages.append({
                "role": "user",
                "content": synthesis_prompt
            })

            # Step 3: Make final synthesis call (text-only, no tools)
            self._trace("🤖 Making final synthesis LLM call (text-only)...", COLOR_YELLOW)
            final_answer = self._llm_call_text_only()

            # Step 4: Validate and return
            if final_answer and final_answer.strip():
                self._trace(f"✅ Synthesis complete - final answer generated ({len(final_answer)} chars)", COLOR_GREEN)
                self._messages.append({"role": "assistant", "content": final_answer})
                return final_answer.strip()
            else:
                # Fallback if synthesis fails
                self._trace("⚠️  WARNING: Synthesis call returned empty response", COLOR_RED)
                fallback_answer = "Unable to generate a final answer based on the gathered information. The investigation completed but synthesis failed."
                self._messages.append({"role": "assistant", "content": fallback_answer})
                return fallback_answer

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            if self.mcp:
                # --- NEW: TRACEABILITY BLOCK ---
                try:
                    # Explicitly fetch the final state of the scratchpad
                    # We send 'content="Final"' just to satisfy the schema, though 'read' ignores it.
                    final_state = await self.mcp.call_tool("ManageJournal", {"action": "read", "content": "Final Trace"})

                    # Print it using the agent's distinct color scheme
                    self._trace(f"🛑 FINAL SCRATCHPAD STATE:\n{final_state}", COLOR_CYAN)
                except Exception:
                    # Fails silently if the server doesn't have the ManageJournal tool yet
                    pass
                # -------------------------------

                await self.mcp.close()
                self._trace("Own MCP server cleanly terminated")

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Executes the actual API call (with tools)."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
            "tools": tools
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(
            f"⏳ Calling {self.model} (timeout: {self.request_timeout}s, max_tokens: {call_params['max_tokens']})", COLOR_CYAN)

        try:
            response = self.client.chat.completions.create(**call_params)
            self._trace(f"✅ LLM call completed successfully", COLOR_GREEN)
            return response
        except Exception as e:
            self._trace(f"❌ LLM call failed: {e}", COLOR_RED)
            raise

    def _llm_call_text_only(self):
        """Executes the actual API call (without tools)."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"⏳ Calling {self.model} (text-only, timeout: {self.request_timeout}s)", COLOR_CYAN)

        try:
            response = self.client.chat.completions.create(**call_params)
            self._trace(f"✅ LLM text-only call completed", COLOR_GREEN)

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"❌ LLM text-only call failed: {e}", COLOR_RED)
            raise

    async def reset(self):
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.tool_call_history = []  # Clear stuck loop detection history
        if self.mcp:
            await self.mcp.close()
            self.mcp = None


if __name__ == "__main__":
    async def run_test():
        agent = KQAProAgent()
        try:
            answer = await agent.ask("Who is the director of Inception?")
            print(f"\n[KQAPro Agent Answer]\n{answer}")
        except Exception as e:
            print(f"Error in test run: {e}")

    asyncio.run(run_test())
