// Dev-only fake backend for `VITE_MOCK=1 npm run dev`. Lets the UI be built
// and screenshotted without the FastAPI server, Qdrant or Virtuoso. It is
// only reachable through a `__MOCK__`-guarded dynamic import in api.ts, so it
// never ships in a production build.
//
// The two SVGs are the real idle figures from lifecycle_svg.py and
// orchestrator_svg.py; the mock lights nodes by rewriting their data-state
// attributes, which is exactly what the server renderer varies.
//
// URL knobs: ?mockSpeed=3 plays runs 3x faster, ?mockPause=14 freezes a run
// at tick 14 (for stable screenshots). A question containing "fail" ends in
// an error event, one containing "limit" gets a 429.
import {
  ApiError,
  type AgentMeta,
  type ApiImpl,
  type DoneEvent,
  type GraphEdge,
  type GraphNode,
  type Meta,
  type RunRecord,
  type Snapshot,
  type TraceEvent,
  type TraceResponse,
} from "../api";
import lifecycleSvg from "./lifecycle.svg?raw";
import orchestratorSvg from "./orchestrator.svg?raw";

const params = new URLSearchParams(window.location.search);
const SPEED = Math.max(0.25, Number(params.get("mockSpeed")) || 1);
const PAUSE = params.has("mockPause") ? Number(params.get("mockPause")) : Infinity;
const TICK_MS = 450 / SPEED;

const AGENTS: AgentMeta[] = [
  {
    name: "Orchestrator (Router)",
    tagline: "Picks the single best specialist for your question, as evaluated in the paper.",
    description: "Routes questions to the best specialist agent (KQAPro or SciQA) automatically.",
    databases: "KQAPro KG + ORKG (via sub-agents)",
    tools: "Delegates to sub-agents",
    orchestrator: true,
    federated: false,
    followups: false,
  },
  {
    name: "Orchestrator (Federated)",
    tagline:
      "Experimental: may ask both specialists in parallel and combine their answers. Slower and uses about twice the tokens; the paper's numbers are single-dispatch.",
    description:
      "Routes questions to one or more specialist agents (KQAPro and/or SciQA); when more than one is picked they run concurrently and their answers are fused into one response.",
    databases: "KQAPro KG + ORKG (via sub-agents)",
    tools: "Delegates to sub-agents",
    orchestrator: true,
    federated: true,
    followups: false,
  },
  {
    name: "KQAPro",
    tagline: "Facts from a general knowledge graph (films, places, people). Use for factual lookups.",
    description: "Answers factual questions over the KQAPro knowledge graph (movies, geography, science facts).",
    databases: "Qdrant (entities/relations) + Virtuoso (SPARQL)",
    tools: "29 MCP tools",
    orchestrator: false,
    federated: false,
    followups: true,
  },
  {
    name: "SciQA",
    tagline: "Scientific research via the ORKG. Use for research papers and contributions.",
    description: "Answers scientific research questions using the Open Research Knowledge Graph (ORKG).",
    databases: "Qdrant (ORKG entities/relations) + Virtuoso (ORKG SPARQL)",
    tools: "27 MCP tools",
    orchestrator: false,
    federated: false,
    followups: true,
  },
];

const META: Meta = {
  title: "AMA-KBQA Assistant",
  agents: AGENTS,
  default_agent: "Orchestrator (Router)",
  suggestions: {
    "Orchestrator (Router)": [
      { label: "Director of Inception", icon: "movie", color: "blue", question: "Who is the director of Inception?" },
      { label: "COVID-19 research", icon: "science", color: "green", question: "What research contributions address COVID-19 detection?" },
      { label: "Einstein's birthplace", icon: "location_on", color: "orange", question: "In which city was Albert Einstein born?" },
    ],
    "Orchestrator (Federated)": [
      {
        label: "Breast cancer, people and papers",
        icon: "hub",
        color: "violet",
        question: "Which notable people have had breast cancer, and what research contributions address breast cancer?",
      },
      {
        label: "Epilepsy and antiepileptic drugs",
        icon: "hub",
        color: "violet",
        question:
          "Which notable people have had epilepsy, and what research contributions address the effectiveness of antiepileptic drugs?",
      },
      { label: "Einstein's birthplace", icon: "location_on", color: "orange", question: "In which city was Albert Einstein born?" },
    ],
    KQAPro: [
      { label: "Director of Inception", icon: "movie", color: "blue", question: "Who is the director of Inception?" },
      { label: "Heavy metal bands like Queen", icon: "music_note", color: "green", question: "How many heavy metal groups are in the genre of Queen?" },
      { label: "Einstein's birthplace", icon: "location_on", color: "orange", question: "In which city was Albert Einstein born?" },
    ],
    SciQA: [
      { label: "COVID-19 research", icon: "science", color: "green", question: "What research contributions address COVID-19 detection?" },
      { label: "Machine learning benchmarks", icon: "biotech", color: "blue", question: "What benchmarks are used for evaluating machine learning models?" },
      { label: "NLP contributions", icon: "article", color: "orange", question: "What are the main contributions in natural language processing?" },
    ],
  },
  models: [
    { id: "kit.deepseek-v4-flash", name: "DeepSeek V4 Flash", price: "$0.14 per 1M in and $0.28 per 1M out" },
    { id: "kit.glm-5.3", name: "GLM-5.3", price: null },
    { id: "kit.mistral-small-4-119b-a8b", name: "Mistral Small 4", price: "$0.10 per 1M in and $0.30 per 1M out" },
  ],
  default_model: "kit.mistral-small-4-119b-a8b",
  default_temperature: 1.0,
  live_graph: true,
  demo: { enabled: false, max_queries_per_session: 20, min_seconds_between_queries: 3 },
  settings: {
    level: "full",
    controls: { model: true, temperature: true, simplified_view: true, live_graph: true },
    endpoints: {
      label: "Endpoints",
      rows: [
        {
          id: "kit-chat",
          label: "KIT KI-Toolbox",
          role: "chat",
          provider: "kit",
          base_url: "https://ki-toolbox.scc.kit.edu/api/v1",
          model: null,
          model_source: "Live /models catalog, chosen in the picker above",
          api_key: { configured: true, hint: "KIT_API_KEY" },
          status: "ok",
          detail: "Chat completions for every agent and sub-agent.",
        },
        {
          id: "kit-embedding",
          label: "KIT KI-Toolbox",
          role: "embedding",
          provider: "kit",
          base_url: "https://ki-toolbox.scc.kit.edu/api/v1",
          model: "kit.qwen3-embedding-8b",
          model_source: "config.toml",
          api_key: { configured: true, hint: "KIT_API_KEY" },
          status: "ok",
          detail: "Embeds your question for vector search over the graphs.",
        },
      ],
      notices: [],
    },
    diagnostics: {
      label: "This build",
      rows: [
        { label: "Retrieval", value: "dense vectors" },
        { label: "Federation", value: "off" },
        { label: "Live graph", value: "on" },
        { label: "Config", value: "config.toml" },
      ],
    },
  },
};

// ── Figures ─────────────────────────────────────────────────────────────────

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function lightUp(svg: string, active: Set<string>, visited: Set<string>, edges: Set<string>, caption: string | null, captionAt: [number, number]): string {
  let out = svg.replace(/data-id="([a-z_]+)" data-state="idle"/g, (_m, id: string) => {
    const state = active.has(id) ? "active" : visited.has(id) ? "visited" : "idle";
    return `data-id="${id}" data-state="${state}"`;
  });
  out = out.replace(
    /data-edge-id="([a-z_]+)" d="([^"]*)" marker-end="url\(#lifecycle-arrowhead\)"/g,
    (m, id: string, d: string) =>
      edges.has(id) ? `data-edge-id="${id}" data-state="active" d="${d}" marker-end="url(#lifecycle-arrowhead-active)"` : m,
  );
  if (caption) {
    out = out.replace(
      "</svg>",
      `<text class="lifecycle-caption" x="${captionAt[0]}" y="${captionAt[1]}" text-anchor="middle">${esc(caption)}</text></svg>`,
    );
  }
  return out;
}

const lifecycle = (a: Set<string>, v: Set<string>, e: Set<string>, caption: string | null) =>
  lightUp(lifecycleSvg, a, v, e, caption, [500, 354]);
const orchestrator = (a: Set<string>, v: Set<string>, e: Set<string>, caption: string | null) =>
  lightUp(orchestratorSvg, a, v, e, caption, [390, 374]);

const ALL_LIFECYCLE = [
  "agent_invocation",
  "pre_classifier",
  "pre_extractor",
  "pre_strategy_inject",
  "main_llm_reason",
  "main_tool_call",
  "main_scratchpad",
  "main_done",
  "post_synthesis",
  "post_evaluate",
  "post_lessons",
];

// ── Scripted run ────────────────────────────────────────────────────────────

interface Step {
  stage: string;
  node: string;
  edge?: string;
  log: string;
  reveal?: number;
}

const C = {
  blue: "#58a6ff",
  green: "#3fb950",
  yellow: "#d29922",
  magenta: "#bc8cff",
  cyan: "#39c5cf",
  grey: "#6e7681",
};
const span = (color: string, text: string) => `<span style="color:${color}">${esc(text)}</span>`;
const bold = (text: string) => `<span style="font-weight:bold; color: #fff;">${esc(text)}</span>`;
const tag = (who: string) => span(C.cyan, `[${who}]`);

const STEPS: Step[] = [
  { stage: "agent_run · ask", node: "agent_invocation", log: `${tag("KQAPro")} ${bold("agent_run")} question received\n` },
  { stage: "classify · question type", node: "pre_classifier", edge: "agent_invocation__pre_classifier", log: `${tag("KQAPro")} question type: ${span(C.yellow, "who / single entity")}\n` },
  { stage: "llm_call · entity extraction", node: "pre_extractor", edge: "pre_classifier__pre_extractor", log: `${tag("KQAPro")} entities: ${span(C.green, "Inception")}  relations: ${span(C.green, "director")}\n` },
  { stage: "strategy injection", node: "pre_strategy_inject", edge: "pre_extractor__pre_strategy_inject", log: `${tag("KQAPro")} strategy: resolve film, then follow ${span(C.magenta, "director")}\n` },
  { stage: "llm_call · reasoning", node: "main_llm_reason", edge: "pre_strategy_inject__main_llm_reason", log: `${tag("KQAPro")} ${bold("llm_call")} iteration 1\n` },
  { stage: "tool_call · SearchEntity", node: "main_tool_call", edge: "main_llm_reason__main_tool_call", reveal: 3, log: `${tag("KQAPro")} ${bold("tool_call")} SearchEntity(query="Inception")\n${span(C.grey, "  -> 3 candidates: Inception (Q25188), Inception (soundtrack), Inception Point")}\n` },
  { stage: "scratchpad", node: "main_scratchpad", edge: "main_tool_call__main_scratchpad", log: `${tag("KQAPro")} scratchpad: film = Q25188\n` },
  { stage: "done? no", node: "main_done", edge: "main_scratchpad__main_done", log: `${tag("KQAPro")} not done, 1 open sub-goal\n` },
  { stage: "llm_call · reasoning", node: "main_llm_reason", edge: "loop_back", log: `${tag("KQAPro")} ${bold("llm_call")} iteration 2\n` },
  { stage: "tool_call · FindNode", node: "main_tool_call", edge: "main_llm_reason__main_tool_call", reveal: 7, log: `${tag("KQAPro")} ${bold("tool_call")} FindNode(id="Q25188")\n${span(C.grey, "  -> director, publication date, cast member, duration")}\n` },
  { stage: "scratchpad", node: "main_scratchpad", edge: "main_tool_call__main_scratchpad", log: `${tag("KQAPro")} scratchpad: director = Q25191 (Christopher Nolan)\n` },
  { stage: "done? no", node: "main_done", edge: "main_scratchpad__main_done", log: `${tag("KQAPro")} verifying with a second relation\n` },
  { stage: "llm_call · reasoning", node: "main_llm_reason", edge: "loop_back", log: `${tag("KQAPro")} ${bold("llm_call")} iteration 3\n` },
  { stage: "tool_call · GetRelations", node: "main_tool_call", edge: "main_llm_reason__main_tool_call", reveal: 12, log: `${tag("KQAPro")} ${bold("tool_call")} GetRelations(entity="Q25191")\n${span(C.grey, "  -> place of birth, spouse, founded")}\n` },
  { stage: "done? yes", node: "main_done", edge: "main_scratchpad__main_done", log: `${tag("KQAPro")} ${span(C.green, "done")}: evidence is sufficient\n` },
  { stage: "synthesis · answer", node: "post_synthesis", edge: "main_done__post_synthesis", reveal: 14, log: `${tag("KQAPro")} ${bold("synthesis")} writing the answer\n` },
  { stage: "trace evaluation", node: "post_evaluate", edge: "post_synthesis__post_evaluate", log: `${tag("KQAPro")} trace evaluation: grounded\n` },
  { stage: "lessons learned", node: "post_lessons", edge: "post_evaluate__post_lessons", log: `${tag("KQAPro")} lessons: none new\n` },
];

const ORCH_PRE: { stage: string; node: string; edge?: string; log: string }[] = [
  { stage: "agent_run · orchestrator", node: "orch_user", log: `${span(C.blue, "[orchestrator]")} question received\n` },
  { stage: "classify · datasource probing", node: "orch_probe", log: `${span(C.blue, "[orchestrator]")} probing datasources: ${span(C.green, "kqapro 0.91")} ${span(C.grey, "sciqa 0.07")}\n` },
  { stage: "delegate · kqapro_agent", node: "orch_dispatch", edge: "dispatch_kqapro", log: `${span(C.blue, "[orchestrator]")} dispatching to ${bold("kqapro_agent")}\n` },
];

const TICKS_PER_STEP = 2;

// ── Graph fixture ───────────────────────────────────────────────────────────

type G = Omit<GraphNode, "highlighted" | "new">;
const t = (label: string, id: string, extra?: string) => `<b>${label}</b><br>${id}${extra ? `<br>${extra}` : ""}`;
const NODES: G[] = [
  { id: "Q25188", label: "Inception", title: t("Inception", "Q25188", "<i>instance of</i>: film"), group: "kqapro" },
  { id: "c:soundtrack", label: "Inception (soundtrack)", title: "<b>Inception (soundtrack)</b><br><i>search candidate</i>", group: "candidate" },
  { id: "c:point", label: "Inception Point", title: "<b>Inception Point</b><br><i>search candidate</i>", group: "candidate" },
  { id: "Q25191", label: "Christopher Nolan", title: t("Christopher Nolan", "Q25191", "<i>occupation</i>: film director"), group: "kqapro" },
  { id: "lit:2010", label: "2010", title: "<i>publication date</i>: 2010-07-16", group: "literal" },
  { id: "Q38111", label: "Leonardo DiCaprio", title: t("Leonardo DiCaprio", "Q38111"), group: "kqapro" },
  { id: "lit:148", label: "148 minutes", title: "<i>duration</i>: 148 minutes", group: "literal" },
  { id: "Q1363", label: "Emma Thomas", title: t("Emma Thomas", "Q1363", "<i>occupation</i>: film producer"), group: "kqapro" },
  { id: "Q126399", label: "Warner Bros.", title: t("Warner Bros.", "Q126399"), group: "kqapro" },
  { id: "Q76364", label: "Hans Zimmer", title: t("Hans Zimmer", "Q76364"), group: "kqapro" },
  { id: "Q2296", label: "Syncopy", title: t("Syncopy", "Q2296", "<i>instance of</i>: production company"), group: "kqapro" },
  { id: "Q84", label: "London", title: t("London", "Q84"), group: "kqapro" },
  { id: "Q163872", label: "The Dark Knight", title: t("The Dark Knight", "Q163872"), group: "kqapro" },
  { id: "Q63985561", label: "Tenet", title: t("Tenet", "Q63985561"), group: "kqapro" },
];
const EDGES: GraphEdge[] = [
  ["Q25188", "Q25191", "director"],
  ["Q25188", "lit:2010", "publication date"],
  ["Q25188", "Q38111", "cast member"],
  ["Q25188", "lit:148", "duration"],
  ["Q25188", "Q1363", "producer"],
  ["Q25188", "Q126399", "distributed by"],
  ["Q25188", "Q76364", "composer"],
  ["Q25188", "Q2296", "production company"],
  ["Q25191", "Q84", "place of birth"],
  ["Q1363", "Q25191", "spouse"],
  ["Q2296", "Q25191", "founded by"],
  ["Q163872", "Q25191", "director"],
  ["Q63985561", "Q25191", "director"],
].map(([from, to, label]) => ({ id: `${from}|${label}|${to}`, from, to, label, title: `${esc(label)}<br><i>FindNode</i>` }));
const HIGHLIGHT = ["Q25188", "Q25191", "Q1363", "Q2296", "lit:148"];

const ANSWER = `**Christopher Nolan** directed *Inception* (2010).

The knowledge graph links the film (Q25188) to him through the \`director\` relation, and records a few related facts:

| Relation | Value |
| --- | --- |
| producer | Emma Thomas |
| production company | Syncopy (founded by Nolan) |
| duration | 148 minutes |`;

function graphAt(count: number, group: "kqapro" | "sciqa", prevCount: number) {
  const shown = NODES.slice(0, count).map((n) => ({
    ...n,
    group: n.group === "kqapro" ? group : n.group,
    highlighted: false,
    new: NODES.indexOf(n) >= prevCount,
  }));
  const ids = new Set(shown.map((n) => n.id));
  const edges = EDGES.filter((e) => ids.has(e.from) && ids.has(e.to));
  return { nodes: shown, edges, stats: { nodes: shown.length, edges: edges.length, truncated: 0 } };
}

// ── Trace fixture ───────────────────────────────────────────────────────────

function makeTrace(question: string, startMs: number, durationS: number, orchestrated: boolean): TraceResponse {
  const ns = (ms: number) => Math.round((startMs + ms) * 1e6);
  const events: TraceEvent[] = [];
  let n = 0;
  const add = (
    parent: string | null,
    kind: string,
    name: string,
    at: number,
    dur: number,
    attributes: Record<string, unknown> = {},
    payload: Record<string, unknown> = {},
    isEvent = false,
  ) => {
    const id = `span${String(++n).padStart(4, "0")}abcd`;
    events.push({
      trace_id: "mocktrace0001",
      span_id: id,
      parent_span_id: parent,
      kind,
      name,
      start_time_unix_nano: ns(at),
      end_time_unix_nano: ns(at + dur),
      duration_ms: dur,
      status: "ok",
      is_event: isEvent,
      attributes,
      payload,
      error: null,
    });
    return id;
  };
  const total = durationS * 1000;
  let root: string | null = null;
  let base = 0;
  if (orchestrated) {
    root = add(null, "agent_run", "Orchestrator.ask", 0, total, { agent: "orchestrator", federation: false });
    add(root, "classify", "datasource probing", 20, 1300, { prompt_tokens: 412, completion_tokens: 38, total_tokens: 450 }, {
      messages: [
        { role: "system", content: "You route questions to the knowledge graph that can answer them." },
        { role: "user", content: question },
      ],
      assistant_content: '{"specialists": ["kqapro_agent"]}',
    });
    root = add(root, "delegate", "kqapro_agent", 1400, total - 1900, { target: "kqapro_agent" });
    base = 1450;
  }
  const agent = add(root, "agent_run", "KQAProAgent.ask", base, total - base - 600, { agent: "kqapro", continuation: false });
  add(agent, "llm_call", "entity extraction", base + 60, 900, { prompt_tokens: 380, completion_tokens: 42 }, {
    messages: [
      { role: "system", content: "Extract the entities and relations the question mentions." },
      { role: "user", content: question },
      { role: "assistant", content: '{"entities": ["Inception"], "relations": ["director"]}' },
    ],
  });
  let at = base + 1100;
  const tools = [
    ["SearchEntity", { query: "Inception" }, '[{"id": "Q25188", "label": "Inception", "score": 0.97}, {"id": "Q1210233", "label": "Inception (soundtrack)", "score": 0.81}]'],
    ["FindNode", { id: "Q25188" }, '{"id": "Q25188", "label": "Inception", "edges": {"director": ["Q25191"], "publication date": ["2010-07-16"]}}'],
    ["GetRelations", { entity: "Q25191" }, '{"place of birth": ["Q84"], "spouse": ["Q1363"]}'],
  ] as const;
  tools.forEach(([tool, args, result], i) => {
    add(agent, "tool_loop_iter", `iteration ${i + 1}`, at, 0, {}, {}, true);
    add(agent, "llm_call", "reasoning", at + 10, 1500, { prompt_tokens: 1800 + i * 450, completion_tokens: 64 }, {
      messages: [
        { role: "user", content: question },
        { role: "assistant", content: null, tool_calls: [{ name: tool, arguments: args }] },
      ],
    });
    add(agent, "tool_call", tool, at + 1550, 420, { tool }, { arguments: args, result });
    at += 2600;
  });
  add(agent, "journal_refresh", "journal snapshot", at, 0, { nodes: 14 }, {}, true);
  add(agent, "synthesis", "answer", at + 50, 1600, { prompt_tokens: 2400, completion_tokens: 118 }, {
    messages: [{ role: "user", content: question }, { role: "assistant", content: ANSWER }],
  });

  const tokens = events.reduce((sum, e) => {
    if (!["llm_call", "classify", "synthesis"].includes(e.kind)) return sum;
    const a = e.attributes as { prompt_tokens?: number; completion_tokens?: number };
    return sum + (a.prompt_tokens ?? 0) + (a.completion_tokens ?? 0);
  }, 0);
  const kindCounts: Record<string, number> = {};
  events.forEach((e) => (kindCounts[e.kind] = (kindCounts[e.kind] ?? 0) + 1));
  return {
    trace_id: "mocktrace0001",
    summary: {
      total_duration_ms: total,
      total_tokens: tokens,
      n_llm_calls: events.filter((e) => ["llm_call", "classify", "synthesis"].includes(e.kind)).length,
      n_tool_calls: events.filter((e) => e.kind === "tool_call").length,
      n_errors: 0,
      kind_counts: kindCounts,
    },
    events,
  };
}

// ── State ───────────────────────────────────────────────────────────────────

interface MockRun {
  record: RunRecord;
  startMs: number;
  orchestrated: boolean;
  group: "kqapro" | "sciqa";
  fail: boolean;
  cancelRequested: boolean;
  final: DoneEvent | null;
  trace: TraceResponse | null;
}

const runs = new Map<string, MockRun>();
const persisted = new Map<string, string>(); // session -> "agent|model"

const wait = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

export const mockApi: ApiImpl = {
  async getMeta() {
    await wait(150);
    if (params.has("mockProviders")) {
      // Shape of the provider-aware builds (demo-booth, demo-llamacpp).
      return {
        ...META,
        models: [
          ...META.models.map((m) => ({ ...m, provider: "KIT" })),
          { id: "openrouter:deepseek/deepseek-v3", name: "DeepSeek V3", price: "$0.27 per 1M in and $1.10 per 1M out", provider: "OpenRouter" },
          { id: "llamacpp:qwen3.6-35b", name: "Qwen3.6 35B (local)", price: null, provider: "llama.cpp" },
        ],
        model_notices: ["The local llama.cpp server is not reachable."],
        // The endpoint rows those branches add through meta.endpoint_rows():
        // a provider without a key, and a local server that is down. Both
        // render from the same component as the KIT rows.
        settings: {
          level: "full",
          controls: { model: true, temperature: true, simplified_view: true, live_graph: true },
          endpoints: {
            label: "Endpoints",
            rows: [
              ...(META.settings?.endpoints?.rows ?? []),
              {
                id: "openrouter-chat",
                label: "OpenRouter",
                role: "chat",
                provider: "openrouter",
                base_url: "https://openrouter.ai/api/v1",
                model: null,
                model_source: "Live /models catalog",
                api_key: { configured: false, hint: "OPENROUTER_API_KEY" },
                status: "no_key",
                detail: "Set OPENROUTER_API_KEY in the environment to reach this endpoint.",
              },
              {
                id: "llamacpp-chat",
                label: "llama-server (chat)",
                role: "chat",
                provider: "llamacpp",
                base_url: "http://host.docker.internal:8080/v1",
                model: "qwen3.6-35b",
                model_source: "whatever the server has loaded",
                api_key: null,
                status: "unreachable",
                detail: "Start llama-server on :8080 to use the local model.",
              },
            ],
            notices: [
              "OpenRouter: no API key configured.",
              "llama-server (chat): not reachable right now.",
            ],
          },
          diagnostics: META.settings?.diagnostics ?? null,
        },
      };
    }
    return META;
  },

  async createRun(body) {
    await wait(120);
    if (!AGENTS.some((a) => a.name === body.agent)) throw new ApiError(400, `Unknown agent: ${body.agent}`);
    if (/limit/i.test(body.question)) {
      throw new ApiError(429, "Demo limit reached: 20 queries per session. Refresh the page to start a new session.");
    }
    for (const r of runs.values()) {
      if (r.record.session_id === body.session_id && r.record.status === "running") {
        throw new ApiError(409, "This session already has a run in flight.");
      }
    }
    const agent = AGENTS.find((a) => a.name === body.agent)!;
    const key = `${body.agent}|${body.model}`;
    // Same rule as the backend: a specialist keeps one agent per session,
    // keyed by (agent, model); the orchestrator always starts fresh.
    const continuation = agent.followups && persisted.get(body.session_id) === key;
    if (agent.followups) persisted.set(body.session_id, key);
    else persisted.delete(body.session_id);
    const id = `mock-${Math.random().toString(36).slice(2, 10)}`;
    runs.set(id, {
      record: {
        run_id: id,
        session_id: body.session_id,
        agent: body.agent,
        question: body.question,
        status: "running",
        started_at: Date.now() / 1000,
      },
      startMs: Date.now(),
      orchestrated: agent.orchestrator,
      group: body.agent === "SciQA" ? "sciqa" : "kqapro",
      fail: /fail/i.test(body.question),
      cancelRequested: false,
      final: null,
      trace: null,
    });
    return { run_id: id, continuation };
  },

  async cancelRun(runId) {
    await wait(60);
    const r = runs.get(runId);
    // Like the server: flip the flag and return. The scripted run notices on
    // its next tick, so the stop is never instantaneous here either.
    if (r && r.record.status === "running") r.cancelRequested = true;
  },

  async resetSession(sessionId) {
    await wait(60);
    persisted.delete(sessionId);
  },

  async getRun(runId) {
    await wait(80);
    const r = runs.get(runId);
    if (!r) throw new ApiError(404, "Unknown run");
    return { ...r.record, ...(r.final ?? {}), status: r.record.status } as RunRecord;
  },

  async getTrace(runId) {
    await wait(200);
    const r = runs.get(runId);
    if (!r) throw new ApiError(404, "Unknown run");
    if (r.record.status === "running") throw new ApiError(409, "The run is still in progress.");
    if (!r.trace) r.trace = makeTrace(r.record.question, r.startMs, r.final?.duration_s ?? 12, r.orchestrated);
    return r.trace;
  },

  subscribeRun(runId, h) {
    const r = runs.get(runId);
    if (!r) {
      window.setTimeout(() => h.onLost("Unknown run"), 50);
      return () => {};
    }
    if (r.final) {
      const final = r.final;
      window.setTimeout(() => h.onDone(final), 50);
      return () => {};
    }

    type PlanStep = { stage: string; node: string; edge?: string; log: string; reveal?: number; sub: boolean };
    const plan: PlanStep[] = [
      ...(r.orchestrated ? ORCH_PRE.map((p) => ({ ...p, sub: false })) : []),
      ...STEPS.map((s) => ({ ...s, sub: true })),
      ...(r.orchestrated
        ? [{ stage: "answer combination", node: "orch_combine", edge: "return_kqapro", log: `${span(C.blue, "[orchestrator]")} combining 1 answer\n`, sub: false }]
        : []),
    ];
    const totalTicks = plan.length * TICKS_PER_STEP;
    let tick = 0;
    let seq = 0;
    let log = "";
    let lastLogLen = -1;
    let revealed = 0;
    let lastReveal = -1;
    const subVisited = new Set<string>();
    const orchVisited = new Set<string>();
    let cancelled = false;

    const phaseOf = (node: string) => (node.startsWith("main") ? "main" : node.startsWith("post") ? "post" : "pre");

    const step = () => {
      if (cancelled) return;
      if (r.cancelRequested) {
        cancelled = true;
        r.record.status = "cancelled";
        h.onCancelled({
          run_id: runId,
          status: "cancelled",
          answer: "Run cancelled.",
          duration_s: (Date.now() - r.startMs) / 1000,
          log_html: log,
        });
        return;
      }
      if (tick >= PAUSE) return;
      const idx = Math.min(plan.length - 1, Math.floor(tick / TICKS_PER_STEP));
      const p = plan[idx];
      const firstTickOfStep = tick % TICKS_PER_STEP === 0;
      if (firstTickOfStep) {
        log += p.log;
        if (p.reveal) revealed = p.reveal;
      }
      const elapsed = (Date.now() - r.startMs) / 1000;

      if (r.fail && idx >= 6) {
        cancelled = true;
        r.record.status = "error";
        h.onError({
          run_id: runId,
          status: "error",
          message: "MCP tool server 'kqapro' closed the connection (simulated failure).",
          log_html: `${log}${span("#ff7b72", "Traceback (most recent call last):\n  ...\nConnectionResetError: MCP server closed the connection")}\n`,
        });
        return;
      }

      const subActive = new Set<string>();
      const subEdges = new Set<string>();
      const orchActive = new Set<string>();
      const orchEdges = new Set<string>();
      if (p.sub) {
        subActive.add(p.node);
        if (p.edge) subEdges.add(p.edge);
        subVisited.add(p.node);
        if (r.orchestrated) {
          orchActive.add("sub_kqapro");
          orchActive.add(`sub_kqapro_${phaseOf(p.node)}`);
          orchVisited.add("sub_kqapro");
          orchVisited.add(`sub_kqapro_${phaseOf(p.node)}`);
        }
      } else {
        orchActive.add(p.node);
        orchVisited.add(p.node);
        if (p.edge) orchEdges.add(p.edge);
      }

      const subSvg = lifecycle(subActive, subVisited, subEdges, r.orchestrated ? null : p.stage);
      const snap: Snapshot = {
        seq: ++seq,
        status: "running",
        stage: p.stage,
        elapsed_s: elapsed,
        span_count: 1 + idx * 3,
        figure: r.orchestrated
          ? { kind: "orchestrator", svg: orchestrator(orchActive, orchVisited, orchEdges, p.stage) }
          : { kind: "lifecycle", svg: subSvg },
        subagents:
          r.orchestrated && (p.sub || idx >= ORCH_PRE.length)
            ? [{ id: "kqapro_agent", display: "KQAPro", status: idx >= ORCH_PRE.length + STEPS.length ? "done" : "running", svg: subSvg }]
            : [],
        log_html: log.length !== lastLogLen ? log : null,
        graph: null,
      };
      lastLogLen = log.length;
      if (revealed !== lastReveal) {
        const g = graphAt(revealed, r.group, Math.max(0, lastReveal));
        snap.graph = { ...g, new_ids: g.nodes.filter((n) => n.new).map((n) => n.id) };
        lastReveal = revealed;
      }
      h.onSnapshot(snap);

      tick += 1;
      if (tick >= totalTicks) {
        const duration = (Date.now() - r.startMs) / 1000;
        const caption = `completed in ${duration.toFixed(1)}s`;
        const g = graphAt(NODES.length, r.group, NODES.length);
        const hl = new Set(HIGHLIGHT);
        const trace = makeTrace(r.record.question, r.startMs, duration, r.orchestrated);
        r.trace = trace;
        const allSub = new Set(ALL_LIFECYCLE);
        const final: DoneEvent = {
          run_id: runId,
          status: "done",
          agent: r.record.agent,
          question: r.record.question,
          answer: ANSWER,
          duration_s: duration,
          tokens: { prompt: 7680, completion: 406, total: 8086 },
          cost: "$0.0009",
          model: "kit.mistral-small-4-119b-a8b",
          trace_id: "mocktrace0001",
          log_html: log,
          figure: r.orchestrated
            ? {
                kind: "orchestrator",
                svg: orchestrator(
                  new Set(),
                  new Set(["orch_user", "orch_probe", "orch_dispatch", "orch_combine", "sub_kqapro", "sub_kqapro_pre", "sub_kqapro_main", "sub_kqapro_post"]),
                  new Set(),
                  caption,
                ),
              }
            : { kind: "lifecycle", svg: lifecycle(new Set(), allSub, new Set(), caption) },
          subagents: r.orchestrated
            ? [{ id: "kqapro_agent", display: "KQAPro", status: "done", svg: lifecycle(new Set(), allSub, new Set(), null) }]
            : [],
          graph: {
            nodes: g.nodes.map((n) => ({ ...n, new: false, highlighted: hl.has(n.id) })),
            edges: g.edges,
            stats: g.stats,
            highlight: HIGHLIGHT,
            answer_nodes: NODES.filter((n) => hl.has(n.id))
              .map((n) => n.label)
              .sort(),
          },
        };
        r.final = final;
        r.record.status = "done";
        cancelled = true;
        h.onDone(final);
        return;
      }
      window.setTimeout(step, TICK_MS);
    };

    window.setTimeout(step, 120);
    return () => {
      cancelled = true;
    };
  },
};
