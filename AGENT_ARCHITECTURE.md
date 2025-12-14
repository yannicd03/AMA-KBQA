# AMA KBQA Agent Architecture

Comprehensive guide to the KQAPro Agent's internal architecture, including pre/post-agent hooks, loop detection mechanisms, and the learning/judge/fewshot example system.

---

## Table of Contents

1. [Agent Overview](#agent-overview)
2. [Pre-Agent Hook System](#pre-agent-hook-system)
3. [Main Agent Loop](#main-agent-loop)
4. [Multi-Layered Loop Detection](#multi-layered-loop-detection)
5. [Post-Agent Hook (Deterministic Synthesis)](#post-agent-hook-deterministic-synthesis)
6. [Learning System: Judge & Few-Shot Examples](#learning-system-judge--few-shot-examples)
7. [Token Tracking](#token-tracking)
8. [State Management](#state-management)
9. [Configuration](#configuration)

---

## Agent Overview

**File:** `ama_kbqa/agents/kqapro_agent/agent.py`

The KQAProAgent is an iterative tool-calling agent that uses:
- **LLM:** Configured via `config.toml` (supports OpenRouter, LMStudio, KIT Ollama, DeepSeek, z.ai)
- **MCP Server:** Loads tools from `ama_kbqa/server/kqapro_server.py`
- **Message History:** Maintains OpenAI-formatted conversation with system prompt, user queries, tool calls, and results
- **Scratchpad (Journal):** Tracks progress, visited nodes, discovered values, and reasoning steps

### Agent Lifecycle

```
1. Initialize agent with LLM client and system prompt
2. Receive user question
3. PRE-AGENT HOOK: Classify question + extract entities
4. MAIN LOOP: Iterative tool calling until answer found
   - LLM decides which tools to call
   - Execute tools and append results to message history
   - Loop detection checks for infinite patterns
   - Periodic journal refreshes for working memory
5. POST-AGENT HOOK: Deterministic synthesis of final answer
6. Reset agent state (for batch processing)
```

---

## Pre-Agent Hook System

**Location:** `ama_kbqa/agents/kqapro_agent/agent.py:717-801`

### Purpose

The pre-agent hook ALWAYS runs BEFORE the main tool-calling loop to provide foundational analysis and strategic guidance.

### What It Does

**Step 1: Run QtypePrediction and EntityExtraction**

Calls two MCP tools in the pre-hook phase:

```python
qtype_result = await self.mcp.call_tool("QtypePrediction", {"question": query})
entities_result = await self.mcp.call_tool("EntityExtraction", {"question": query})
```

**Step 2: Parse Results**

Extracts:
- `qtype` - Question type classification (Count, Verify, QueryAttr, SelectBetween, etc.)
- `fewshot_examples` - Curated examples for this question type
- `entities` - Extracted entity names
- `relations` - Extracted relation names

**Step 3: Load Question-Type-Specific Strategy**

The agent has hard-coded reasoning strategies for each question type stored in `KQAProAgent.QTYPE_STRATEGIES`:

```python
QTYPE_STRATEGIES = {
    "Count": "...",       # Strategy for aggregation questions
    "Verify": "...",      # Strategy for boolean questions
    "QueryAttr": "...",   # Strategy for attribute lookups
    # ... etc
}
```

Each strategy includes:
- **Topology Description:** Graph structure pattern for this question type
- **Decision Tree:** Step-by-step approach for solving
- **SPARQL Patterns:** Example queries
- **Tool Recommendations:** Which tools work best
- **Edge Cases:** Common pitfalls and solutions

**Step 4: Inject Analysis Context**

Creates a formatted analysis message and injects it as a user message:

```
═══════════════════════════════════════════════════════════════════════
PRE-ANALYSIS (Automatically Computed)
═══════════════════════════════════════════════════════════════════════

Question Type: Count

Extracted Entities:
  - Inception
  - Christopher Nolan

Extracted Relations:
  - directed by

[TYPE-SPECIFIC STRATEGY GUIDE]

[OPTIONAL: FEW-SHOT EXAMPLES]
═══════════════════════════════════════════════════════════════════════
```

### Benefits

1. **Deterministic Classification:** All questions get classified BEFORE reasoning begins
2. **Strategic Guidance:** LLM knows optimal approach for each question type
3. **Few-Shot Learning:** Relevant examples guide the agent's tool selection
4. **Entity Awareness:** Pre-extracted entities help focus the search
5. **Consistent Reasoning:** Every question gets the same foundational analysis

### Configuration

Pre-agent hook uses:
- **QtypePrediction tool:** Classifies question using LLM in JSON mode
- **EntityExtraction tool:** Extracts entities/relations using LLM in JSON mode
- **Few-shot examples:** Loaded from `db/datasets/kqapro/fewshot-examples/`

---

## Main Agent Loop

**Location:** `ama_kbqa/agents/kqapro_agent/agent.py:803-1047`

### Loop Structure

```python
iteration_count = 0
max_iterations = 50

while True:
    iteration_count += 1

    # Safety check
    if iteration_count > max_iterations:
        return "Error: Agent reached maximum iteration limit"

    # Periodic journal refresh (every 5 iterations)
    if iteration_count % 5 == 0:
        journal_refresh = await self.mcp.call_tool("GetJournalSummary", {})
        # Inject journal as user message
        # Check for progress (compare with last journal state)

    # Call LLM with tools
    response = self._llm_call(tools=openai_tools)
    message = response.choices[0].message

    # Update token usage
    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
    # ... etc

    # If no tool calls, break to synthesis
    if not message.tool_calls:
        break

    # Add assistant message to history
    self._messages.append(msg_dict)

    # Execute each tool call
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # LOOP DETECTION
        loop_detected, loop_reason = self._detect_loops(func_name, func_args)

        if loop_detected:
            # Inject intervention message
            tool_result = "⚠️ INFINITE LOOP DETECTED ..."
        else:
            # Normal execution
            tool_result = await self.mcp.call_tool(func_name, func_args)

        # Append tool result to history
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": tool_result
        })

    # If GetJournalSummary was called, inject answer prompt
    if called_get_journal_summary:
        self._messages.append({
            "role": "user",
            "content": "You MUST provide your final answer now. Do NOT call any more tools."
        })
```

### Key Features

**1. Safety Limits**
- Max 50 iterations to prevent infinite loops
- Warnings at iteration 45
- Periodic status updates every 10 iterations

**2. Periodic Journal Refreshes**
- Every 5 iterations, inject GetJournalSummary into context
- Provides "working memory refresh" so agent doesn't forget discoveries
- Includes progress check (see loop detection below)

**3. Token Usage Tracking**
- Tracks `prompt_tokens`, `completion_tokens`, `total_tokens`
- Updated after every LLM call
- Reported in batch results

**4. Color-Coded Tracing**
- BLUE - General messages
- GREEN - Tool calls/results
- RED - Errors
- YELLOW - Warnings
- CYAN - Important state changes

**5. GetJournalSummary Auto-Answer Trigger**
- If agent calls GetJournalSummary, inject a user prompt forcing final answer
- Prevents agent from calling more tools after reviewing journal

---

## Multi-Layered Loop Detection

**Location:** `ama_kbqa/agents/kqapro_agent/agent.py:528-673`

### Overview

The agent implements **4 layers** of loop detection to catch different infinite loop patterns.

### Detection Mechanisms

#### 1. Identical Repeated Calls

**Pattern:** Same tool with same parameters called 3 times in a row

**Detection Code:**
```python
if len(self.tool_call_history) >= 3:
    recent_calls = self.tool_call_history[-3:]
    if all(call == current_call for call in recent_calls):
        return True, f"Identical call repeated 3 times: {func_name}(...)"
```

**Example:**
- FindNode("Boston")
- FindNode("Boston")
- FindNode("Boston") → **LOOP DETECTED**

---

#### 2. Oscillating Pattern (A-B-A-B)

**Pattern:** Agent alternates between 2 or 3 tools repeatedly

**Detection Code:**
```python
# 2-tool oscillation (A-B-A-B-A-B)
if last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5]:
    return True, f"Oscillating between {last_6[0]} and {last_6[1]}"

# 3-tool oscillation (A-B-C-A-B-C)
if last_6[0] == last_6[3] and last_6[1] == last_6[4] and last_6[2] == last_6[5]:
    return True, f"Oscillating between {last_6[0]}, {last_6[1]}, {last_6[2]}"
```

**Example:**
- GetAttributeDetails
- FindNode
- GetAttributeDetails
- FindNode
- GetAttributeDetails
- FindNode → **LOOP DETECTED** (A-B-A-B-A-B pattern)

---

#### 3. Tool Spam (Same Tool 5+ Times in 6 Calls)

**Pattern:** Agent overuses a single tool even with different parameters

**Detection Code:**
```python
if len(self.tool_sequence) >= 6:
    last_6 = self.tool_sequence[-6:]
    for tool, count in tool_counts.items():
        if count >= 5:
            return True, f"Tool '{tool}' called {count} times in last 6 iterations"
```

**Example:**
- RunSPARQL (query 1)
- RunSPARQL (query 2)
- GetAttributeDetails
- RunSPARQL (query 3)
- RunSPARQL (query 4)
- RunSPARQL (query 5) → **LOOP DETECTED** (RunSPARQL called 5/6 times)

---

#### 4. No Progress Detection (Journal Unchanged)

**Pattern:** Journal state hasn't changed in 5 iterations

**Detection Code:**
```python
if self.last_journal_state is not None and journal_refresh == self.last_journal_state:
    # Inject stronger intervention
    self._messages.append({
        "role": "user",
        "content": "⚠️ **NO PROGRESS DETECTED** ..."
    })
```

**Checked every 5 iterations** during periodic journal refresh

**Example:**
- Iteration 10: Journal has 3 nodes, 2 values
- Iteration 15: Journal STILL has 3 nodes, 2 values → **NO PROGRESS WARNING**

---

### Loop Recovery System

When a loop is detected, the system:

**1. Injects Tool-Specific Guidance**

Each tool has customized recovery instructions:

```python
def _get_tool_specific_loop_guidance(self, func_name: str) -> str:
    guidance_map = {
        "RunSPARQL": "STOP using RunSPARQL - Use simpler tools...",
        "GetAttributeDetails": "Check available_attributes from FindNode...",
        "FindNode": "Search for a related entity instead...",
        # ... etc
    }
```

**2. Fetches Journal Summary**

Provides agent with current state:
```
⚠️ INFINITE LOOP DETECTED

Detected Pattern: Oscillating between FindNode and GetAttributeDetails

Your current approach is not making progress. You MUST change strategy.

Required Actions:
1. STOP using 'FindNode' - it's not working
2. Review what you've already discovered (see below)
3. Try a FUNDAMENTALLY different approach:

[TOOL-SPECIFIC GUIDANCE]

═══════════════════════════════════════════════════════════════════════
📋 WHAT YOU'VE ALREADY DISCOVERED:
═══════════════════════════════════════════════════════════════════════

[JOURNAL SUMMARY]

Based on the above, formulate a DIFFERENT strategy or acknowledge if data doesn't exist.
```

**3. Clears Recent History**

```python
self.tool_call_history = []
self.tool_sequence = []
```

Allows agent to make fresh attempts without being penalized

---

### Loop Prevention Best Practices

**Agent Behavior:**
- Checks journal before revisiting nodes
- Uses GetNodeSummary instead of repeated GetAttributeDetails
- Pivots to different search strategies after 2 failures
- Acknowledges when data doesn't exist

**System Support:**
- Auto-updates journal to track visited_nodes
- Periodic journal refreshes remind agent of discoveries
- Tool-specific guidance suggests alternatives
- Progress detection catches subtle loops

---

## Post-Agent Hook (Deterministic Synthesis)

**Location:** `ama_kbqa/agents/kqapro_agent/agent.py:1048-1103`

### Purpose

Ensures EVERY question gets a complete, well-formed final answer based on gathered data.

### Why It Exists

**Problem:** Agent might:
- Get stuck in loops before formulating answer
- Call tools but never synthesize findings
- Provide incomplete or unclear answers

**Solution:** Deterministic synthesis step ALWAYS runs after tool gathering completes

### How It Works

**Step 1: Detect Loop Exit**

When LLM returns message without `tool_calls`, break from main loop

**Step 2: Fetch Complete Journal**

```python
journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
```

Gets ALL discovered information:
- Visited nodes
- Found values (MOST IMPORTANT)
- Verified facts
- Completed steps

**Step 3: Inject Synthesis Prompt**

```python
synthesis_prompt = f"""You have completed your tool-based investigation. Here is EVERYTHING you discovered:

═══════════════════════════════════════════════════════════════════════
JOURNAL SUMMARY - ALL DISCOVERED INFORMATION
═══════════════════════════════════════════════════════════════════════

{journal_summary}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════

Based ONLY on the information shown above in your journal, provide a clear, direct, and complete answer to:

"{query}"

INSTRUCTIONS:
- Use ONLY the facts and values from your journal summary above
- Provide a direct answer without explaining your process
- If the information is insufficient, state exactly what is missing
- Be concise but complete

YOUR FINAL ANSWER:"""
```

**Step 4: Make Final LLM Call (Text-Only)**

```python
final_answer = self._llm_call_text_only()
```

LLM generates answer WITHOUT access to tools (prevents infinite loops)

**Step 5: Validate and Return**

```python
if final_answer and final_answer.strip():
    return final_answer.strip()
else:
    return "Unable to generate a final answer based on the gathered information."
```

### Benefits

1. **Guaranteed Termination:** Always produces an answer (or explicit failure message)
2. **Journal-Based:** Answer is grounded in discovered data, not hallucination
3. **Concise:** Forces LLM to synthesize without rambling
4. **Traceable:** Journal shows exactly what was used to form answer

---

## Learning System: Judge & Few-Shot Examples

### Overview

The system implements a **continuous learning loop** where:
1. Agent answers questions
2. Judge evaluates answers
3. High-quality examples are exported as few-shot learning data
4. Future agents use these examples for better performance

### Architecture

```
┌─────────────────┐
│  Batch Runner   │  Runs N questions through agent
└────────┬────────┘
         │
         v
┌─────────────────┐
│  LLM Judge      │  Evaluates each answer
│  (Optional)     │  - Correctness
└────────┬────────┘  - Reasoning quality
         │           - Argumentation score (1-5)
         │           - Improvement suggestions
         v
┌─────────────────┐
│  Export Filter  │  Filters high-quality examples
│                 │  - Correct answers only
└────────┬────────┘  - Argumentation score >= 4
         │
         v
┌─────────────────┐
│  Few-Shot Files │  Saved to fewshot-examples/
│  (by qtype)     │  - Count.json
└────────┬────────┘  - Verify.json
         │           - Query.json, etc.
         │
         v
┌─────────────────┐
│  Pre-Agent Hook │  Loads examples during classification
│  (QtypePrediction)│ Injects into agent context
└─────────────────┘
```

---

### 1. Batch Runner

**File:** `ama_kbqa/agents/kqapro_agent/batch_runner.py`

**Purpose:** Process multiple questions and collect detailed results

**Usage:**
```bash
# Run with LLM judge (evaluates correctness and quality)
python ama_kbqa/agents/kqapro_agent/batch_runner.py --n_questions 10 --seed 42 --postprocessing_mode llm_judge

# Run with SPARQL verification only (faster, no quality assessment)
python ama_kbqa/agents/kqapro_agent/batch_runner.py --n_questions 10 --seed 42 --postprocessing_mode sparql
```

**What It Does:**
1. Samples N random questions from validation dataset
2. Runs each question through KQAProAgent
3. Captures console output, token usage, duration, tool calls
4. Optionally calls LLM Judge for evaluation
5. Saves results to `batch_results/BatchXXX/`
6. Exports high-quality examples to few-shot database

**Output Structure:**
```
batch_results/
├── Batch001/
│   ├── sampled_questions.json  # Questions selected
│   ├── results.json            # Per-question results with metadata
│   ├── summary.json            # Accuracy and statistics
│   └── console_logs/           # Full console output per question
│       ├── question_001.txt
│       └── ...
```

---

### 2. LLM Judge

**Configuration:** `config.toml` under `[postprocessing]` section

```toml
[postprocessing]
judge_provider = "openrouter"
judge_model = "deepseek/deepseek-v3.2-speciale"
judge_temperature = 0.0
chat_model_provider = ""  # Optional: Provider-specific routing
```

**Purpose:** Evaluate answer correctness, reasoning quality, and provide improvement suggestions

**Structured Output (Pydantic Model):**
```python
class AnswerJudgment(BaseModel):
    is_correct: bool  # Semantic equivalence to gold answer
    correctness_reasoning: str  # Detailed comparison
    argumentation_quality: str  # Assessment of reasoning structure
    argumentation_score: int  # 1-5 score (1=very poor, 5=excellent)
    suggested_improvement: str  # Actionable feedback
```

**Judge Prompt:**
```
You are an expert evaluator for Knowledge Base Question Answering systems.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Evaluate:
1. Correctness (semantic equivalence)
2. Argumentation quality (logical reasoning, sound steps)
3. Specific improvements for future similar questions

Provide structured assessment.
```

**Benefits:**
- **Semantic Matching:** Judges semantic equivalence, not exact string match
- **Quality Assessment:** Scores reasoning quality (1-5)
- **Learning Feedback:** Provides actionable improvement suggestions
- **Automated Filtering:** High scores (≥4) automatically become few-shot examples

---

### 3. Few-Shot Example Export

**File:** `ama_kbqa/agents/kqapro_agent/batch_runner.py:136-231`

**Function:** `export_fewshot_examples_from_judgments()`

**What It Does:**
1. Filters results for correct answers with argumentation_score ≥ 4
2. Groups examples by question type (qtype)
3. Creates example entry:
   ```json
   {
     "question": "...",
     "reasoning": "...",  # From correctness_reasoning
     "correct_qtype": "Query",
     "lesson_learned": "...",  # From suggested_improvement
     "argumentation_score": 5,
     "exported_at": "2025-12-14T01:19:25.584546"
   }
   ```
4. Merges with existing examples (no duplicates)
5. Sorts by argumentation_score (highest first)
6. Limits to top 10 examples per qtype
7. Saves to `db/datasets/kqapro/fewshot-examples/{qtype}.json`

**Example Files:**
- `Count.json` - Aggregation questions
- `Verify.json` - Boolean questions
- `Query.json` - General factual questions
- `SelectBetween.json` - Binary comparisons
- `SelectAmong.json` - Superlatives
- etc.

---

### 4. Few-Shot Loading System

**File:** `ama_kbqa/server/kqapro_server.py:486-573`

**Function:** `load_fewshot_examples(max_per_type=10, specific_qtype=None)`

**What It Does:**
1. Loads examples from `fewshot-examples/` directory
2. Limits to `max_per_type` examples per question type
3. Formats for prompt injection:
   ```
   ### Few-Shot Examples (Curated from Past Classifications)

   **Example 1:**
   Question: What is the ISO 3166-2 code of the city with NUTS code UKE11?
   Correct Type: Query
   Reasoning: The agent correctly identified the ISO code...
   Lesson Learned: Could improve by mentioning city name explicitly...
   ```

**When It's Used:**
- Called by **QtypePrediction** tool during pre-agent hook
- Examples loaded AFTER classification (only loads examples for predicted qtype)
- Injected into pre-analysis context for agent guidance

---

### 5. Continuous Learning Loop

**How It Works:**

**Iteration 1: Initial Run**
- Agent answers questions using only hard-coded strategies
- No few-shot examples available
- Judge evaluates answers
- High-quality examples exported

**Iteration 2: Learning Phase**
- New agent loads few-shot examples from Iteration 1
- Examples guide tool selection and reasoning
- Performance improves
- New high-quality examples exported (different questions)

**Iteration N: Steady State**
- System has 10 high-quality examples per qtype
- Examples are continuously refined (new better examples replace old)
- Agent performance stabilizes

**Self-Improvement Mechanism:**
- High argumentation scores (4-5) → Examples exported
- Low argumentation scores (1-3) → Filtered out
- Examples sorted by score → Best examples used first
- Duplicate questions prevented → Diverse example set

---

### 6. Configuration

**Judge Configuration (`config.toml`):**
```toml
[postprocessing]
judge_provider = "openrouter"          # LLM provider for judging
judge_model = "deepseek/deepseek-v3.2-speciale"  # Judge model
judge_temperature = 0.0                 # Low temp for consistent judgments
chat_model_provider = ""                # Optional provider routing
```

**Example Directory:**
- `db/datasets/kqapro/fewshot-examples/`
- JSON files per question type
- Template: `_example_template.json`

**Quality Thresholds:**
- Minimum argumentation score: 4 (good) or 5 (excellent)
- Maximum examples per type: 10
- Duplicate prevention: By question text

---

## Token Tracking

**Location:** `ama_kbqa/agents/kqapro_agent/agent.py:373-378, 889-892, 1175-1178`

### Tracked Metrics

```python
self.token_usage = {
    "prompt_tokens": 0,      # Input to LLM
    "completion_tokens": 0,  # Output from LLM
    "total_tokens": 0        # Sum of above
}
```

### When Updated

**1. After every LLM call with tools:**
```python
if response.usage:
    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
    self.token_usage["completion_tokens"] += response.usage.completion_tokens
    self.token_usage["total_tokens"] += response.usage.total_tokens
```

**2. After text-only synthesis call:**
```python
if response.usage:
    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
    # ... (same as above)
```

### Accumulation

- Tokens accumulate across ALL tool-calling iterations
- Includes both main loop and synthesis step
- Tracked separately for each question in batch processing
- Reported in batch results JSON

### Batch Results Include

```json
{
  "prompt_tokens": 15234,
  "completion_tokens": 856,
  "total_tokens": 16090,
  "duration_seconds": 12.45,
  "agent_turns": 8
}
```

---

## State Management

### Message History

**Format:** OpenAI-compatible conversation format

```python
self._messages = [
    {"role": "system", "content": self.system_prompt},
    {"role": "user", "content": "What is the population of Boston?"},
    {"role": "user", "content": "[PRE-ANALYSIS CONTEXT]"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "FindNode", "content": "..."},
    # ... more iterations
    {"role": "user", "content": "[SYNTHESIS PROMPT]"},
    {"role": "assistant", "content": "The population of Boston is ..."}
]
```

### Journal (Scratchpad)

**Global State:** `session_journal` in `kqapro_server.py`

**Structure (Pydantic Model):**
```python
class JournalState(BaseModel):
    question_text: str
    question_type: str
    target_entities: list[str]
    target_attributes: list[str]
    visited_nodes: dict[str, str]  # {node_id: node_name}
    verified_facts: list[dict]
    failed_attempts: list[str]
    found_values: dict[str, dict[str, Any]]  # {entity_id: {attr: value}}
    current_plan: list[str]
    completed_steps: list[str]
    partial_answer: str
```

**Auto-Updated By:**
- `FindNode` → Updates `visited_nodes`
- `GetAttributeDetails` → Updates `found_values`
- Tools → Update `verified_facts`, `failed_attempts`

**Manually Updated Via:**
- `ManageJournal` tool (rare)
- Pre-agent hook sets `question_type`

### Reset Between Questions

**Batch Processing:**
```python
await agent.reset()
```

**What Gets Reset:**
- Message history (except system prompt)
- Token usage counters
- Loop detection tracking (tool_call_history, tool_sequence)
- Journal state (via MCP server restart)

**What Persists:**
- LLM client configuration
- MCP server connection (closed and reopened)
- System prompt

---

## Configuration

### LLM Provider Configuration

**File:** `config.toml`

```toml
[llm]
chat_provider = "openrouter"      # Main agent LLM
embedding_provider = "openrouter"  # Vector search embeddings

chat_temperature = 0.2
chat_max_tokens = 15000

[openrouter]
base_url = "https://openrouter.ai/api/v1"
chat_model = "minimax/minimax-m2"
embedding_model = "qwen/qwen3-embedding-8b"
chat_model_provider = "minimax/fp8"  # Optional routing
```

**Supported Providers:**
- OpenRouter (multiple cloud models)
- LMStudio (local models)
- KIT Ollama (KIT-specific endpoint)
- DeepSeek (DeepSeek API)
- z.ai (GLM models)

### Agent Parameters

**File:** `ama_kbqa/agents/kqapro_agent/agent.py`

**Configurable:**
- `REQUEST_TIMEOUT_SECONDS` (default: 60) - LLM call timeout
- `max_iterations` (default: 50) - Loop iteration limit
- Journal refresh interval (default: every 5 iterations)
- Loop detection thresholds (3 identical calls, 5/6 tool spam)

**System Prompt:**
- Hard-coded in `__init__` method
- Includes KBQA-specific rules, tool descriptions, execution guidelines
- Question-type strategies injected via QTYPE_STRATEGIES dictionary

---

## Execution Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
│  - Load LLM client from config.toml                             │
│  - Initialize MCP server connection                             │
│  - Load system prompt with KBQA guidelines                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│ 2. PRE-AGENT HOOK (Deterministic Classification)               │
│  - QtypePrediction: Classify question type                      │
│  - EntityExtraction: Extract entities/relations                 │
│  - Load question-type-specific strategy                         │
│  - Load curated few-shot examples for qtype                     │
│  - Inject pre-analysis context into message history             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│ 3. MAIN AGENT LOOP (Iterative Tool Calling)                    │
│  REPEAT until answer found OR max iterations:                   │
│   - Call LLM with tools                                         │
│   - Track token usage                                           │
│   - Execute tool calls                                          │
│   - LOOP DETECTION: Check for infinite patterns                 │
│   - If loop detected: Inject intervention message               │
│   - Every 5 iterations: Inject journal refresh                  │
│   - PROGRESS CHECK: Compare journal state                       │
│   - If GetJournalSummary called: Force answer next turn         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│ 4. POST-AGENT HOOK (Deterministic Synthesis)                   │
│  - Fetch complete GetJournalSummary                             │
│  - Inject synthesis prompt with all discovered data             │
│  - Make final LLM call (text-only, no tools)                    │
│  - Validate and return final answer                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│ 5. OPTIONAL: LLM JUDGE (Batch Processing Only)                 │
│  - Evaluate answer correctness (semantic match)                 │
│  - Score argumentation quality (1-5)                            │
│  - Provide improvement suggestions                              │
│  - Export high-quality examples (score ≥ 4)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│ 6. RESET & NEXT QUESTION (Batch Mode)                          │
│  - Clear message history                                        │
│  - Reset token counters                                         │
│  - Clear loop detection state                                   │
│  - Close and restart MCP server (resets journal)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Determinism Where Possible
- Pre-agent hook ALWAYS runs
- Post-agent synthesis ALWAYS runs
- Classification happens BEFORE reasoning
- Final answer ALWAYS grounded in journal

### 2. Multi-Layered Safety
- Loop detection catches 4 different patterns
- Progress tracking via journal comparison
- Max iteration limits
- Tool-specific recovery guidance

### 3. Continuous Learning
- High-quality examples automatically exported
- Few-shot learning improves future performance
- Judge provides actionable feedback
- Example diversity via duplicate prevention

### 4. Transparency & Traceability
- Color-coded console output
- Complete message history preserved
- Journal tracks all discoveries
- Token usage and timing metrics

### 5. Configurability
- LLM providers via config.toml
- Judge model separate from agent model
- Adjustable thresholds and timeouts
- Modular tool system via MCP

---

## Common Patterns

### Adding New Question Types

1. Define strategy in `QTYPE_STRATEGIES` dictionary
2. Add to `QtypePredictionResponse` Literal type
3. Create few-shot example file: `{qtype}.json`
4. Update `_build_classification_prompt()` if needed

### Debugging Agent Behavior

1. Check console logs for colored trace output
2. Review `batch_results/BatchXXX/console_logs/` for full execution
3. Examine journal state at different iterations
4. Look for loop detection warnings
5. Check token usage for unexpected spikes

### Improving Agent Performance

1. Run batch with LLM judge enabled
2. Review `suggested_improvement` fields in judgments
3. Export high-quality examples to few-shot database
4. Run new batch to validate improvement
5. Iterate: judge → export → improve

---

## Future Enhancements

**Potential Improvements:**

1. **Adaptive Loop Detection**
   - Learn loop patterns from past failures
   - Dynamic threshold adjustment

2. **Tool Usage Analytics**
   - Track which tools work best for each qtype
   - Recommend tool sequences

3. **Multi-Turn Reasoning**
   - Break complex questions into sub-questions
   - Hierarchical planning

4. **Error Recovery Strategies**
   - Learned fallback patterns
   - Automatic strategy switching

5. **Real-Time Learning**
   - Online few-shot example updates
   - Incremental model fine-tuning
