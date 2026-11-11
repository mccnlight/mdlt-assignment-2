"""Baseline: a big chat LLM asked to return JSON (OpenAI-compatible chat API).

Providers (LLM_PROVIDER):
  cloudflare - Workers AI, free daily allocation           [default]
  openrouter - OpenRouter (free models capped at 50 requests/day)
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "cloudflare")
_DEFAULT_MODEL = {"cloudflare": "@cf/openai/gpt-oss-120b", "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free"}
LLM_MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODEL[LLM_PROVIDER])
USD_PER_1K_NEURONS = 0.011  # Workers AI list price


class DailyLimitError(RuntimeError):
    """Quota exhausted until tomorrow; retrying is pointless."""


def _endpoint():
    if LLM_PROVIDER == "cloudflare":
        return (f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}"
                "/ai/v1/chat/completions", os.environ["CLOUDFLARE_API_TOKEN"])
    return "https://openrouter.ai/api/v1/chat/completions", os.environ["OPENROUTER_API_KEY"]


def chat(system, user, timeout=120):
    """Return (raw_text, parsed_json_or_None, usage, latency_seconds, cost_usd)."""
    url, key = _endpoint()
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if LLM_PROVIDER == "openrouter":
        body["usage"] = {"include": True}
    t0 = time.perf_counter()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
    latency = time.perf_counter() - t0
    if r.status_code == 429 and ("per-day" in r.text or "daily" in r.text.lower()):
        raise DailyLimitError(f"{r.status_code}: {r.text[:300]}")
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    data = r.json()
    if "choices" not in data:
        raise RuntimeError(f"no choices: {json.dumps(data)[:500]}")
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})
    try:
        parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = None
    if "neurons" in usage:
        cost = usage["neurons"] * USD_PER_1K_NEURONS / 1000
    else:
        cost = float(usage.get("cost", 0) or 0)
    return text, parsed, usage, latency, cost


if __name__ == "__main__":
    system = (
        "You route support tickets. Reply with JSON only: "
        '{"department": "billing"|"technical"|"sales", "frustration": 0|1|2, "urgent": true|false}'
    )
    text, parsed, usage, latency, cost = chat(system, "Help! My payouts have been failing for 3 days.")
    print(f"model: {LLM_MODEL}")
    print("raw:", text)
    print("parsed:", parsed)
    print("usage:", usage, f"latency: {latency:.2f}s", f"cost: ${cost:.8f}")
