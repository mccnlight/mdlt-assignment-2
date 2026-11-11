"""Client for System One decision models (state + typed questions -> typed answers).

Providers:
  clef - Cloudflare Clef on Workers AI (free tier)       [default]
  jev  - TypeSafe Jev via Vercel AI Gateway (paid credit only)
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("DECISION_PROVIDER", "clef")
CLEF_MODEL = os.environ.get("CLEF_MODEL", "@cf/cloudflare/clef")

# List price, USD per 1M input tokens; output tokens are free for both models.
# (Our Clef calls ran inside Cloudflare's free daily allocation.)
PRICE_PER_M_INPUT = {"clef": 0.24, "jev": 0.042}


def _call_clef(state, questions, timeout):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}"
        f"/ai/run/{CLEF_MODEL}"
    )
    headers = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}
    r = requests.post(url, headers=headers, json={"state": state, "questions": questions}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    return r.json()["result"]


def _call_jev(state, questions, timeout):
    url = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    headers = {"Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"}
    body = {"model": "typesafe-ai/jev", "state": state, "questions": questions}
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    return r.json()


_CALLERS = {"clef": _call_clef, "jev": _call_jev}


def ask(state, questions, timeout=60):
    """Return (answers, usage, latency_seconds, cost_usd). Raises on HTTP error."""
    t0 = time.perf_counter()
    data = _CALLERS[PROVIDER](state, questions, timeout)
    latency = time.perf_counter() - t0
    usage = data.get("usage", {})
    cost = usage.get("input_tokens", 0) * PRICE_PER_M_INPUT[PROVIDER] / 1e6
    return data["answers"], usage, latency, cost


if __name__ == "__main__":
    import json

    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this message convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
    }
    answers, usage, latency, cost = ask("Help! My payouts have been failing for 3 days.", questions)
    print(f"provider: {PROVIDER}")
    print(json.dumps(answers, indent=2))
    print("usage:", usage, f"latency: {latency:.2f}s", f"cost: ${cost:.8f}")
