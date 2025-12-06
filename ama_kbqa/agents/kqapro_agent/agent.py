from __future__ import annotations
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

load_dotenv(override=True)

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
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "arcee-ai/trinity-mini")
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
                await asyncio.sleep(0.05)
            except (CancelledError, RuntimeError) as e:
                trace(
                    self.agent_name, f"{COLOR_YELLOW}WARNING: MCP-Close error on shutdown ignored ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False


# --- KQAProAgent ---

class KQAProAgent:

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing in .env")

        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )

        self.model = MODEL_NAME
        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        # NEW: Token Tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        self.system_prompt = """SYSTEM ROLE
            You are the KQAPro Execution Agent. Your goal is to answer natural language questions by querying a Knowledge Graph (KG).

            CRITICAL RULES
            1.  No Hallucination: You have NO internal knowledge. You must verify every fact using the tools.
            2.  Schema Compliance: You must use the valid predicates returned by `ExploreNeighborhood`. Do not guess predicate names (e.g., do not guess `wdt:P123`, find it first).
            3.  State Management: You MUST use `ManageJournal` before every other tool call to update your plan.

            KNOWLEDGE GRAPH SPECIFICS (CRITICAL)
            You are operating on a specific ontology. You MUST use the following prefixes in your thought process and SPARQL construction. DO NOT define these in your `RunSPARQL` calls; the server injects them automatically.

            * `ex:` -> Entities (e.g., `ex:Q64`)
            * `prop:` -> Properties/Relations (e.g., `prop:P1082`)
            * `attr:` -> Attributes
            * `qual:` -> Qualifiers
            * `unit:` -> Units

            YOUR WORKING MEMORY (SCRATCHPAD)
            You have access to the tool `ManageJournal`. This is your most important tool.
            You MUST use it at every step to:
            1. Log visited nodes: To ensure you do not run in circles (Loop Avoidance).
            2. Save facts: When you have verified a triple, write it down here.
            3. Planning: Update your plan whenever you find new information.

            EXECUTION LOOP (General Strategy)
            1.Analyze the question (using the QtypePrediction tool): Analyze the question and classify the question as one of the 9 types of Questions mentioned above via the tool.
            2.Analyze (using the EntityExtraction tool): Call EntityExtraction to break the question into Entities and Relations.
            3.Map (using the FindNode tool): specific QIDs for the entities found in Step 1.
            4.Log (using the ManageJournal tool): Log the QIDs, Relations, ... found.
            5.Explore (ExploreNeighborhood): Use the QID and the extracted relation to find the correct predicate/fact.
            6.Refine & Execute: Use your tools and iterate through nodes and relations until you have enough information to answer the question.
            7.Answer: Provide the final answer directly and strictly from the tool outputs.

            REASONING STRATEGIES (THE 9 QUESTION TYPES)

            PROTOCOL FOR ANSWER RETRIEVAL:
            1. Primary Action (Tool Use): Always prioritize standard retrieval tools to answer the user's question.
            2. Secondary Action (SPARQL Fallback): ONLY construct and execute SPARQL queries if the standard tools return ambiguous results, incomplete data, or fail to answer.
            3. Verification Action: If a definitive answer is found via tools, you may optionally use the SPARQL templates below to verify the correctness.

            1. Count (Aggregation)
            Use this logic for questions about counting entities.
            Triggers: "How many...", "Count the number of..."
            Topology: [Entity] -> [Predicate] -> [Target_Nodes]
            Strategy: 
                1. Tools: Attempt to retrieve the count or a list of items using search tools.
                2. SPARQL (Fallback/Verify): Find the Entity QID and Attribute PID. Count distinct targets.
            SPARQL Template:
            SELECT (COUNT(DISTINCT ?target) AS ?count) WHERE {
            ex:ENTITY_ID prop:PREDICATE ?target .
            # Optional: ?target prop:instance_of ex:TARGET_TYPE .
            }

            2. QueryAttr (Direct Lookup)
            Use this logic for questions about retrieving a specific property of an entity.
            Triggers: "What is the [Attribute] of [Entity]?", "Who is the [Relation] of [Entity]?"
            Topology: [Entity] -> [Predicate] -> [Target]
            Strategy:
                1. Tools: Use standard lookup to find the attribute value.
                2. SPARQL (Fallback/Verify): Identify the source QID and relation PID to traverse the graph.
            SPARQL Templates:
            SELECT ?label WHERE {
            ex:ENTITY_ID prop:PREDICATE ?targetEntity .
            ?targetEntity rdfs:label ?label .
            }
            # OR for literals:
            SELECT ?value WHERE {
            ex:ENTITY_ID attr:ATTRIBUTE ?value .
            }

            3. QueryAttrQualifier (Contextual Fact)
            Use this logic for questions asking for an entity dependent on context or a condition.
            Triggers: "When did...", "Where did...", "At what location..."
            Topology: [Entity] <-[is_subject_of]- [Fact_Node] -[has_qualifier]-> [Value]
            Strategy:
                1. Tools: Search for the specific event or context using natural language queries.
                2. SPARQL (Fallback/Verify): Isolate the Fact Node and retrieve the specific qualifier (Time/Location).
            SPARQL Template:
            SELECT ?qualifier_value WHERE {
            # 1. Navigate from Entity to the Fact/Statement Node
            ex:ENTITY_ID prop:PREDICATE ?fact_node .
            # 2. Retrieve the specific qualifier from the Fact Node
            ?fact_node qual:QUALIFIER_PREDICATE ?qualifier_value .
            }

            4. QueryName (Reverse Lookup)
            Use this logic for questions asking for a name based on an object and context.
            Triggers: "Which [Type] has [Attribute]...?", "Who [Action] [Object]?"
            Topology: [Target?] -> [Predicate] -> [Known_Object]
            Strategy:
                1. Tools: Search using the object and relation as keywords to find the subject.
                2. SPARQL (Fallback/Verify): Perform a reverse traversal from the Known Object to find the Subject.
            SPARQL Template:
            SELECT ?subjectLabel WHERE {
            # Reverse traversal: Find ?subject pointing to known OBJECT
            ?subject prop:PREDICATE ex:OBJECT_ID .
            ?subject rdfs:label ?subjectLabel .
            }

            5. QueryRelation (Relationship Identification)
            Use this logic for questions asking for a relation between 2 entities
            Triggers: "How are [A] and [B] related?", "What is the connection between..."
            Topology: [Entity_A] <-> [?Predicate] <-> [Entity_B]
            Strategy:
                1. Tools: Query for the connection between the two entities directly.
                2. SPARQL (Fallback/Verify): Check both directions (A->B or B->A) to find the predicate connecting them.
            SPARQL Template:
            SELECT DISTINCT ?relationLabel WHERE {
            { ex:ENTITY_A_ID ?p ex:ENTITY_B_ID . }
            UNION
            { ex:ENTITY_B_ID ?p ex:ENTITY_A_ID . }
            ?p rdfs:label ?relationLabel .
            }

            6. QueryRelationQualifier (Relation Detail)
            Use this logic for questions asking about details of a specific relation.
            Triggers: "What role...", "In what capacity...", "How precisely..."
            Topology: [Entity_A] <-[in_statement]- [Fact_Node] -[has_role]-> [Role_Value]
            Strategy:
                1. Tools: Search for the nuance or role definition in the relationship between A and B.
                2. SPARQL (Fallback/Verify): Find the intermediate Fact Node connecting A and B, then extract the attribute.
            SPARQL Template:
            SELECT ?detail_value WHERE {
            ex:ENTITY_A_ID prop:PREDICATE ?fact_node .
            ?fact_node prop:target ex:ENTITY_B_ID . 
            ?fact_node qual:DETAIL_PREDICATE ?detail_value .
            }

            7. SelectAmong (Superlative/Sorting)
            Use this logic for questions asking about an entity being in a certain spot in a group.  
            Triggers: "Who is the tallest...", "What is the most recent...", "First...", "Last..."
            Topology: [Group] -> [Member] -> [Value]
            Strategy:
                1. Tools: Search for ranked lists or superlative facts.
                2. SPARQL (Fallback/Verify): Retrieve all group members, sort by the attribute, and limit to the top result.
            SPARQL Template:
            SELECT ?itemLabel WHERE {
            ?item prop:instance_of ex:GROUP_ID .
            ?item attr:SORT_ATTRIBUTE ?value .
            ?item rdfs:label ?itemLabel .
            }
            ORDER BY DESC(?value) # Use ASC(?value) for 'Smallest'/'First'
            LIMIT 1

            8. SelectBetween (Binary Comparison)
            Use this logic for questions comparing 2 entities. 
            Triggers: "Who is older: [A] or [B]?", "Which has more [X]: [A] or [B]?"
            Topology: [Entity_A/B] -> [Attribute] -> [Value]
            Strategy:
                1. Tools: Retrieve the specific attribute for both A and B via tools and compare logically.
                2. SPARQL (Fallback/Verify): Filter strictly to the two specific QIDs, retrieve values, sort, and return the winner.
            SPARQL Template:
            SELECT ?itemLabel WHERE {
            VALUES ?item { ex:ENTITY_A_ID ex:ENTITY_B_ID }
            ?item attr:ATTRIBUTE ?value .
            ?item rdfs:label ?itemLabel .
            }
            ORDER BY DESC(?value) 
            LIMIT 1

            9. Verify (Boolean Check)
            Use this logic for questions asking for binary validation. 
            Triggers: "Is [A] a [B]?", "Does [A] have [Attribute] > [X]?"
            Topology: [Entity] -> [Attribute] ? [Value]
            Strategy:
                1. Tools: Attempt to confirm the fact using search or specific lookup tools.
                2. SPARQL (Fallback/Verify): Use an ASK query to return a definitive True/False.
            SPARQL Template:
            ASK {
            ex:ENTITY_ID attr:ATTRIBUTE ?value .
            FILTER (?value > "TARGET_VALUE"^^xsd:decimal) 
            }

            
            """

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _trace(self, msg: str, color: str = COLOR_GREEN):
        trace(self.name, msg, color)

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

            # 2. Iterative tool-call loop
            self._messages.append({"role": "user", "content": query})

            while True:
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message

                # Update Token Usage
                if response.usage:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens

                # Logging intermediate thoughts (Thoughts/Text)
                if message.content:
                    self._trace(f"🧠 Thought/Text: {message.content}", COLOR_BLUE)

                # Final answer (no more tools)
                if not message.tool_calls:
                    assistant_text = message.content or ""
                    self._messages.append({"role": "assistant", "content": assistant_text})
                    self._trace("🏁 Final answer generated by LLM.")
                    return assistant_text.strip()

                # --- CORRECTION START ---
                # The assistant message MUST be added to the history ONCE
                # BEFORE processing the tools.
                self._messages.append(message)
                # --- CORRECTION END ---

                # Execute tool call(s)
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    # Logging
                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"🛠️  Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    # Execution
                    tool_result = await self.mcp.call_tool(func_name, func_args)

                    # Logging result
                    log_result = tool_result
                    if len(log_result) > 500:
                        log_result = log_result[:500] + f"... [truncated, total len: {len(tool_result)}]"
                    self._trace(f"🔙 Result ({func_name}): {log_result}", COLOR_CYAN)

                    # --- BUG WAS HERE (Removed: self._messages.append(message)) ---

                    # Only the tool result is appended in the loop
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

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
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
            tools=tools
        )

    def _llm_call_text_only(self):
        """Executes the actual API call (without tools)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens

        return response.choices[0].message.content

    async def reset(self):
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
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
