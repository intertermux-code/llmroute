#!/usr/bin/env python3
"""
llmroute — benchmark-driven router for OpenAI-compatible LLM endpoints.

Measure your endpoints once, then route every request by policy:

    llmroute bench --config endpoints.toml
    llmroute run --config endpoints.toml --policy fastest --prompt "summarize this"
    llmroute run --config endpoints.toml --policy cheapest --tag code < prompt.txt

Policies: fastest, cheapest, balanced. Failed endpoints fall through to the
next candidate automatically. Standard library only, Python 3.11+.
"""

import argparse
import json
import os
import statistics
import sys
import time
import tomllib
import urllib.request
import urllib.error
from datetime import datetime, timezone

VERSION = "0.1.0"
POLICIES = ("fastest", "cheapest", "balanced")


class RouteError(Exception):
    pass


def die(msg, code=1):
    print(f"llmroute: error: {msg}", file=sys.stderr)
    sys.exit(code)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- config

def default_config_path():
    for cand in ("./endpoints.toml",
                 os.path.expanduser("~/.config/llmroute/endpoints.toml")):
        if os.path.exists(cand):
            return cand
    return "./endpoints.toml"


def load_config(path):
    if not os.path.exists(path):
        die(f"config not found: {path} (see README for the format)")
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        die(f"invalid TOML in {path}: {e}")
    eps = cfg.get("endpoint")
    if not eps or not isinstance(eps, list):
        die(f"{path}: need at least one [[endpoint]] section")
    for ep in eps:
        for key in ("name", "base_url", "model"):
            if not ep.get(key):
                die(f"{path}: endpoint missing required key {key!r}")
            if not isinstance(ep[key], str):
                die(f"{path}: endpoint key {key!r} must be a string, got {type(ep[key]).__name__}")
        ep["base_url"] = ep["base_url"].rstrip("/")
        ep.setdefault("tags", [])
        ep.setdefault("price_in", None)
        ep.setdefault("price_out", None)
    names = [e["name"] for e in eps]
    if len(set(names)) != len(names):
        die(f"{path}: duplicate endpoint names")
    return eps


def results_path_for(config_path, override=None):
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), "results.json")


def load_results(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data.get("endpoints", {}) if isinstance(data, dict) else {}


def api_key_for(ep):
    env = ep.get("api_key_env")
    if not env:
        return None
    key = os.environ.get(env)
    if not key:
        die(f"endpoint {ep['name']!r}: env var {env} is not set")
    return key


# ---------------------------------------------------------------- http

def chat_complete(ep, messages, max_tokens, timeout):
    """POST /chat/completions. Returns (content, usage, latency_ms) or raises RouteError."""
    url = ep["base_url"] + "/chat/completions"
    body = {"model": ep["model"], "messages": messages,
            "max_tokens": max_tokens, "stream": False}
    headers = {"Content-Type": "application/json"}
    key = api_key_for(ep)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        raise RouteError(f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RouteError(f"connection failed: {e}")
    latency_ms = (time.monotonic() - t0) * 1000
    if status != 200:
        raise RouteError(f"HTTP {status}")
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise RouteError(f"bad response: {e}")
    usage = data.get("usage") or {}
    return content, usage, latency_ms


# ---------------------------------------------------------------- bench

def bench_endpoint(ep, probes, timeout):
    lat, ok = [], 0
    for _ in range(probes):
        try:
            _, _, ms = chat_complete(
                ep, [{"role": "user", "content": "ping"}], 1, timeout)
            lat.append(ms)
            ok += 1
        except RouteError as e:
            print(f"llmroute: bench: {ep['name']}: {e}", file=sys.stderr)
    return {
        "p50_ms": round(statistics.median(lat), 1) if lat else None,
        "success_rate": round(ok / probes, 2),
        "probes": probes,
        "probed_at": now_iso(),
    }


def cmd_bench(args):
    eps = load_config(args.config)
    if args.tag:
        eps = [e for e in eps if args.tag in e.get("tags", [])]
        if not eps:
            die(f"no endpoints tagged {args.tag!r}")
    out_path = results_path_for(args.config, args.results)
    results = {"endpoints": {}, "benched_at": now_iso()}
    for ep in eps:
        print(f"llmroute: benching {ep['name']} ({args.probes} probes)...", file=sys.stderr)
        results["endpoints"][ep["name"]] = bench_endpoint(ep, args.probes, args.timeout)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, out_path)
    for name, st in results["endpoints"].items():
        p50 = f"{st['p50_ms']:.0f}ms" if st["p50_ms"] is not None else "n/a"
        print(f"  {name:<24} p50={p50:<10} success={st['success_rate']:.0%}")
    print(f"llmroute: wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------- routing

def estimate_cost(ep, in_chars, max_tokens):
    """USD estimate. Unknown price -> +inf (sorted last under cheapest)."""
    if ep.get("price_in") is None or ep.get("price_out") is None:
        return float("inf")
    # ponytail: chars/4 token heuristic, named as what it is
    in_tokens = in_chars / 4
    return ep["price_in"] * in_tokens / 1e6 + ep["price_out"] * max_tokens / 1e6


def order_endpoints(eps, stats, policy, in_chars, max_tokens):
    def latency(ep):
        st = stats.get(ep["name"], {})
        ms = st.get("p50_ms")
        # unmeasured or failing endpoints sort last, stable by name
        return (ms is None or st.get("success_rate", 0) == 0, ms or 0, ep["name"])

    if policy == "fastest":
        return sorted(eps, key=latency)
    if policy == "cheapest":
        return sorted(eps, key=lambda e: (estimate_cost(e, in_chars, max_tokens), e["name"]))
    # balanced: mean of latency-rank and cost-rank
    by_lat = {e["name"]: i for i, e in enumerate(sorted(eps, key=latency))}
    by_cost = {e["name"]: i for i, e in enumerate(
        sorted(eps, key=lambda e: (estimate_cost(e, in_chars, max_tokens), e["name"])))}
    return sorted(eps, key=lambda e: ((by_lat[e["name"]] + by_cost[e["name"]]) / 2, e["name"]))


def cmd_run(args):
    eps = load_config(args.config)
    if args.policy not in POLICIES:
        die(f"unknown policy {args.policy!r} (choose from {', '.join(POLICIES)})")
    if args.tag:
        eps = [e for e in eps if args.tag in e.get("tags", [])]
        if not eps:
            die(f"no endpoints tagged {args.tag!r}")
    stats = load_results(results_path_for(args.config, args.results))
    if not stats:
        die("no benchmark results found — run `llmroute bench` first")

    if args.prompt is not None:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        die("no prompt: pass --prompt or pipe stdin")
    if not prompt.strip():
        die("empty prompt")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    ordered = order_endpoints(eps, stats, args.policy, len(prompt) + len(args.system or ""),
                              args.max_tokens)
    if not args.quiet:
        print(f"llmroute: order: {' > '.join(e['name'] for e in ordered)}", file=sys.stderr)
    errors = []
    for ep in ordered:
        t0 = time.monotonic()
        try:
            content, usage, _ = chat_complete(ep, messages, args.max_tokens, args.timeout)
        except RouteError as e:
            errors.append(f"{ep['name']}: {e}")
            print(f"llmroute: {ep['name']} failed ({e}), trying next", file=sys.stderr)
            continue
        dt = time.monotonic() - t0
        if not args.quiet:
            u = ""
            if usage.get("prompt_tokens") is not None:
                u = (f" tokens in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}")
            print(f"llmroute: served by {ep['name']} in {dt:.1f}s{u}", file=sys.stderr)
        print(content)
        return
    die("all endpoints failed:\n  " + "\n  ".join(errors))


def cmd_list(args):
    eps = load_config(args.config)
    stats = load_results(results_path_for(args.config, args.results))
    print(f"{'NAME':<22}{'MODEL':<28}{'P50':<10}{'SUCCESS':<9}PRICE $/1M (in/out)")
    for ep in eps:
        st = stats.get(ep["name"], {})
        p50 = f"{st['p50_ms']:.0f}ms" if st.get("p50_ms") is not None else "-"
        succ = f"{st.get('success_rate', 0):.0%}" if st else "-"
        price = "-"
        if ep.get("price_in") is not None:
            price = f"{ep['price_in']}/{ep['price_out']}"
        tags = f" [{','.join(ep['tags'])}]" if ep["tags"] else ""
        print(f"{ep['name']:<22}{ep['model']:<28}{p50:<10}{succ:<9}{price}{tags}")


# ---------------------------------------------------------------- cli

def build_parser():
    p = argparse.ArgumentParser(prog="llmroute",
                                description="Benchmark-driven router for OpenAI-compatible LLM endpoints.")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("-c", "--config", default=None, help="endpoints TOML (default: ./endpoints.toml or ~/.config/llmroute/endpoints.toml)")
    p.add_argument("--results", default=None, help="results JSON path (default: next to the config)")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="benchmark every endpoint, save results")
    b.add_argument("--probes", type=int, default=3, help="probes per endpoint (default: 3)")
    b.add_argument("--timeout", type=float, default=30)
    b.add_argument("--tag", default=None, help="only bench endpoints with this tag")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("run", help="route one prompt by policy, with fallback")
    r.add_argument("--policy", default="fastest", choices=POLICIES)
    r.add_argument("--tag", default=None)
    r.add_argument("--prompt", default=None)
    r.add_argument("--system", default=None)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("list", help="show endpoints and benchmark stats")
    l.set_defaults(func=cmd_list)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.config is None:
        args.config = default_config_path()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
