"""Support-ticket routing for a digital bank: typed questions + threshold rules.

ticket --> Choice(department) + Score(urgency) + Noul(escalate) --> route
"""
import decision_client
import llm_client

DEPARTMENTS = ["payments", "technical", "security", "general"]
PRIORITIES = ["P3", "P2", "P1"]  # index = urgency level

# Catch-all option is last: System One models lean toward the first Choice option.
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team of a digital bank should handle this customer support ticket?",
        "criteria": {
            "payments": "Transfers, payouts, card payments, refunds, fees, limits, declined or stuck money",
            "technical": "App crashes, login or OTP problems, errors, bugs, slow or broken features",
            "security": "Fraud, unauthorized transactions, stolen or lost card, hacked account, phishing, scams",
            "general": "Questions about products, tariffs, opening accounts, documents, feedback, anything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How quickly does the bank need to act on this ticket?",
        "criteria": [
            "Can wait a few days: a question, suggestion or minor inconvenience",
            "Should be handled today: a problem that blocks the customer but no money is at risk",
            "Needs action within an hour: money is being lost, stolen or blocked right now",
        ],
    },
    "escalate": {
        "type": "noul",
        "instructions": "Should a senior agent take this ticket because the customer is very upset?",
        "criteria": {
            "true": "Customer threatens to leave, complain to a regulator or go to court, or demands a manager",
            "false": "Customer is calm or mildly annoyed and makes none of these threats or demands",
        },
    },
}

ESCALATE_THRESHOLD = 0.5


def route(department, priority):
    # Security is always handled as top priority.
    return "security/P1" if department == "security" else f"{department}/{priority}"


def decide_system_one(ticket):
    """Return a decision dict from the System One model."""
    answers, usage, latency, cost = decision_client.ask(ticket, QUESTIONS)
    dep = answers["department"]
    urg = answers["urgency"]
    # Use the most likely urgency level, not the averaged score (score is weak at magnitudes).
    level = int(max(urg["probabilities"], key=urg["probabilities"].get))
    p_dep = dep["probabilities"][dep["choice"]]
    p_level = urg["probabilities"][str(level)]
    department = dep["choice"]
    confidence = p_dep if department == "security" else min(p_dep, p_level)
    return {
        "department": department,
        "priority": PRIORITIES[level],
        "escalate": answers["escalate"]["noul"] >= ESCALATE_THRESHOLD,
        "route": route(department, PRIORITIES[level]),
        "confidence": confidence,
        "p_escalate": answers["escalate"]["noul"],
        "valid": True,
        "latency": latency,
        "cost": cost,
        "input_tokens": usage.get("input_tokens", 0),
        "raw": answers,
    }


LLM_SYSTEM = f"""You route customer support tickets for a digital bank in Kazakhstan.
Tickets may be in English, Russian or Kazakh. Answer with a single JSON object and nothing else:
{{"department": one of {DEPARTMENTS}, "urgency": 0 | 1 | 2, "escalate": true | false}}

department:
""" + "\n".join(f"- {k}: {v}" for k, v in QUESTIONS["department"]["criteria"].items()) + """

urgency:
""" + "\n".join(f"- {i}: {v}" for i, v in enumerate(QUESTIONS["urgency"]["criteria"])) + f"""

escalate:
- true: {QUESTIONS["escalate"]["criteria"]["true"]}
- false: {QUESTIONS["escalate"]["criteria"]["false"]}"""


def decide_llm(ticket):
    """Return a decision dict from the LLM baseline; valid=False if the JSON breaks the schema."""
    text, parsed, usage, latency, cost = llm_client.chat(LLM_SYSTEM, ticket)
    out = {"latency": latency, "cost": cost, "raw": text, "valid": False, "confidence": None,
           "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
    if not isinstance(parsed, dict):
        return out
    dep, urg, esc = parsed.get("department"), parsed.get("urgency"), parsed.get("escalate")
    if dep not in DEPARTMENTS or urg not in (0, 1, 2) or not isinstance(esc, bool):
        return out
    out.update(department=dep, priority=PRIORITIES[urg], escalate=esc,
               route=route(dep, PRIORITIES[urg]), valid=True)
    return out


if __name__ == "__main__":
    samples = [
        "Someone just withdrew 200 000 tenge from my card in Almaty, I am in Astana! Block it NOW",
        "Приложение вылетает при входе после обновления, не могу проверить баланс",
        "What documents do I need to open a deposit account?",
        "Kaspi-ге аударым жасадым, ақша 2 күн бойы келмей жатыр. Fix it or I'm closing my account and writing to the regulator!",
    ]
    for t in samples:
        s = decide_system_one(t)
        print(f"\n{t}\n  S1 : {s['route']:<14} esc={s['escalate']} conf={s['confidence']:.2f} {s['latency']:.2f}s")
        l = decide_llm(t)
        print(f"  LLM: {l.get('route', 'INVALID'):<14} esc={l.get('escalate')} {l['latency']:.2f}s")
