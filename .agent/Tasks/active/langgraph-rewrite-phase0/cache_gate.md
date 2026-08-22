# Prefix-cache gate (KIT endpoint), raw openai client vs LangGraph/ChatOpenAI

Model: `kit.gemma4-31b-it` | trials per case: 3 | total requests: 18

Base prompt: the real captured 15-tool, 4-message QueryAttr request body (`scratchpad/spike/old_request.json`), byte-identical across both paths. Each COLD trial prepends a unique nonce to the system message (position 0 — the most cache-defeating spot). WARM immediately repeats that exact prompt. GROWN appends one assistant tool-call + tool-result pair to the warmed prompt.


## raw

| case | trial | ttft (s) | prompt_tokens |
|---|---|---|---|
| cold | 0 | 16.39 | 5214 |
| cold | 1 | 18.13 | 5213 |
| cold | 2 | 21.01 | 5213 |
| warm | 0 | 17.12 | 5214 |
| warm | 1 | 18.29 | 5213 |
| warm | 2 | 12.77 | 5213 |
| grown | 0 | 19.26 | 5257 |
| grown | 1 | 12.36 | 5256 |
| grown | 2 | 20.96 | 5256 |

median TTFT: cold=18.13s warm=17.12s grown=19.26s
warm/cold ratio: 0.94
grown/cold ratio: 1.06

## langgraph

| case | trial | ttft (s) | prompt_tokens |
|---|---|---|---|
| cold | 0 | 14.81 | 5215 |
| cold | 1 | 13.48 | 5215 |
| cold | 2 | 6.87 | 5215 |
| warm | 0 | 18.81 | 5215 |
| warm | 1 | 11.00 | 5215 |
| warm | 2 | 7.71 | 5215 |
| grown | 0 | 12.04 | 5258 |
| grown | 1 | 6.70 | 5258 |
| grown | 2 | 7.42 | 5258 |

median TTFT: cold=13.48s warm=11.00s grown=7.42s
warm/cold ratio: 0.82
grown/cold ratio: 0.55

## raw vs langgraph, side by side (median TTFT, seconds)

| case | raw | langgraph |
|---|---|---|
| cold | 18.13 | 13.48 |
| warm | 17.12 | 11.00 |
| grown | 19.26 | 7.42 |
