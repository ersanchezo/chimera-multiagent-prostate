"""
explanation.py
Agent 8: Explanation — patient-friendly summary with RAG context.
"""

import json
import yaml

from langchain_openai import ChatOpenAI

from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

EXPLANATION_PROMPT = """You are a patient communication agent.
Generate a clear, compassionate explanation of the biopsy decision.

Decision: {decision}
Ensemble score: {score}
Key factors: {key_factors}
Reasoning summary: {summary}
Guideline validation: {verdict}

Write in plain language, avoiding medical jargon. Explain:
1. What the findings mean
2. Why the biopsy decision was made
3. What happens next

Keep it under 200 words.
"""


def _get_llm():
    return ChatOpenAI(
        model=_CFG["llm"]["model"],
        base_url=_CFG["llm"]["base_url"],
        api_key=_CFG["llm"]["api_key"],
        temperature=0.3,
        max_tokens=512,
    )


def explanation_agent(state: dict) -> dict:
    decision = state.get("biopsy_decision") or {}
    fusion = state.get("fusion_result") or {}
    validation = state.get("validation_result") or {}

    prompt = EXPLANATION_PROMPT.format(
        decision=fusion.get("final_recommendation", "uncertain"),
        score=fusion.get("ensemble_score", "N/A"),
        key_factors=", ".join(decision.get("key_factors", [])),
        summary=decision.get("summary", ""),
        verdict=validation.get("final_verdict", "N/A"),
    )

    try:
        response = _get_llm().invoke(prompt)
        explanation = response.content
    except Exception as e:
        explanation = f"Explanation unavailable: {str(e)}"

    return {
        "explanation": explanation,
        "audit_log": log_entry(
            state, "explanation", "generated_explanation",
            {"length": len(explanation)},
        ),
    }