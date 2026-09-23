"""
guideline_validator.py
Agent 6: Guideline Reasoning Validator — RAG-grounded validation.
"""

import json
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from schemas.schemas import GuidelineValidation
from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

VALIDATOR_PROMPT = """You are a clinical guideline and ML consistency validator.

=== DECISION TO VALIDATE ===
Decision: {decision}
Confidence: {confidence}
Reasoning chain:
{reasoning_chain}
Key factors: {key_factors}

=== ML MODEL OUTPUTS ===
MRI model csPCa probability: {mri_prob} (confidence {mri_conf})
Clinical readiness score: {clin_prob} (confidence {clin_conf})

=== RETRIEVED GUIDELINE EVIDENCE ===
{evidence}

=== TASK ===
Perform three checks:

CHECK 1 — GUIDELINE CONSISTENCY:
For each reasoning step, verify it is supported by the retrieved guidelines.

CHECK 2 — ML AGREEMENT:
Compare the LLM decision with ML model outputs.
- If LLM says "yes" but both models < 0.3, flag disagreement.
- If LLM says "no" but both models > 0.7, flag disagreement.
Weight disagreements by model confidence.

CHECK 3 — INTERNAL CONSISTENCY:
Verify assigned weights match reasoning content.

Produce a verdict and, if applicable, a correction.

{format_instructions}
"""


def _get_llm():
    return ChatOpenAI(
        model=_CFG["llm"]["model"],
        base_url=_CFG["llm"]["base_url"],
        api_key=_CFG["llm"]["api_key"],
        temperature=0.0,
        max_tokens=2048,
    )


def guideline_validator_agent(state: dict) -> dict:
    decision = state.get("biopsy_decision") or {}
    mri = state.get("mri_prediction") or {}
    clin = state.get("clinical_prediction") or {}
    evidence = state.get("retrieved_evidence") or []

    parser = PydanticOutputParser(pydantic_object=GuidelineValidation)

    evidence_text = "\n\n".join([
        f"[Source: {e.get('source', 'unknown')}, Page {e.get('page', '?')}, "
        f"Relevance: {e.get('score', 'N/A')}]\n{e.get('content', '')}"
        for e in evidence[:10]
    ])

    prompt = VALIDATOR_PROMPT.format(
        decision=decision.get("final_decision", "unknown"),
        confidence=decision.get("confidence", "unknown"),
        reasoning_chain=json.dumps(decision.get("reasoning_chain", []), indent=2),
        key_factors=", ".join(decision.get("key_factors", [])),
        mri_prob=mri.get("csPCa_probability", "N/A"),
        mri_conf=mri.get("confidence", "N/A"),
        clin_prob=clin.get("readiness_score", "N/A"),
        clin_conf=clin.get("confidence", "N/A"),
        evidence=evidence_text,
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = _get_llm().invoke(prompt)
        validation = parser.parse(response.content)
    except Exception as e:
        validation = GuidelineValidation(
            is_reasoning_consistent=False,
            consistency_score=0.0,
            guideline_consistent=False,
            ml_agreement=False,
            unsupported_claims=[f"Validation parsing error: {str(e)}"],
            final_verdict="REJECTED",
        )

    should_accept = (
        validation.final_verdict in ("VALIDATED", "PARTIALLY_VALIDATED")
        and validation.consistency_score >= 0.6
    )

    return {
        "validation_result": validation.model_dump(),
        "decision_accepted": should_accept,
        "audit_log": log_entry(
            state, "guideline_validator", "validated_reasoning",
            {"verdict": validation.final_verdict,
             "score": validation.consistency_score,
             "accepted": should_accept},
        ),
    }