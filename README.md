# llmroute

[![ci](https://github.com/intertermux-code/llmroute/actions/workflows/ci.yml/badge.svg)](https://github.com/intertermux-code/llmroute/actions)

Benchmark-driven router for OpenAI-compatible LLM endpoints. If you juggle
several models, gateways, or API keys, you already have a routing table —
it's just in your head, or in a stale doc. llmroute measures your endpoints
and routes every request by policy, with automatic fallback when one fails.

```sh
pip install llmroute

# describe your endpoints once
cat > endpoints.toml <<'EOF'
[[endpoint]]
name = "fast-small"
base_url = "https://gateway.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
model = "small-8b"
price_in = 0.20    # USD per 1M tokens
price_out = 0.40
tags = ["parallel"]

[[endpoint]]
name = "big-brain"
base_url = "https://gateway.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
model = "large-400b"
price_in = 3.00
price_out = 9.00
tags = ["serial"]
EOF

llmroute bench --config endpoints.toml
llmroute list --config endpoints.toml

llmroute run --config endpoints.toml --policy fastest  --prompt "summarize this thread"
llmroute run --config endpoints.toml --policy cheapest --tag parallel < draft.md
```

Zero dependencies, standard library only. Python 3.11+.

## Policies

- **fastest** — lowest benchmarked p50 latency first.
- **cheapest** — lowest estimated cost first, from your `price_in`/`price_out`
  and a documented chars/4 token heuristic. Endpoints without prices sort last.
- **balanced** — mean of latency rank and cost rank.

Endpoints that error, time out, or return garbage are skipped with a stderr
note and the next candidate is tried. If everything fails, you get the full
list of errors and a non-zero exit. `run` prints the response body to stdout
and routing metadata (`served by X in Ys, tokens …`) to stderr, so it composes
with pipes. `--quiet` silences the metadata.

`bench` sends a few tiny `ping` probes per endpoint and stores p50 latency
plus success rate in `results.json` next to the config (override with
`--results`). Re-bench whenever your lineup changes; `run` refuses to guess
without fresh numbers.

## Why not just use LiteLLM?

LiteLLM is a full proxy server you deploy and operate. llmroute is a
300-line CLI you call per request — no server, no database, no config
service. Different tool for a different job: pick this when you want routing
as a shell primitive, not infrastructure.

## Limitations

- Routing is by measured latency, price, and availability — not by answer
  quality. Tag your strong models and select them with `--tag` when quality
  is what matters.
- Non-streaming `/chat/completions` only. If you need tokens as they arrive,
  this isn't your tool.

## License

MIT.
