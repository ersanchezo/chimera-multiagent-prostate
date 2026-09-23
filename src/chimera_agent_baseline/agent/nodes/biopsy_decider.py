"""
biopsy_decider.py
Agent 5: Biopsy Decider — CoT reasoning with ML scores + guideline evidence.
"""

import json
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from schemas.schemas import BiopsyDecisionOutput
from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

BIOPSY_DECIDER_PROMPT = """You are a urology biopsy decision agent.
Decide whether a prostate biopsy is indicated.

=== PATIENT FEATURES ===
{features}

=== ML MODEL PREDICTIONS ===
MRI embedding model (csPCa probability): {mri_prob}
  (model confidence: {mri_conf})
Clinical readiness model (biopsy warranted probability): {clin_prob}
  (model confidence: {clin_conf})

Top contributing features from clinical model:
{clin_contributions}

=== RISK CATEGORY ===
{risk_category}

=== RETRIEVED GUIDELINE EVIDENCE ===
{evidence}

=== REASONING GUIDANCE ===
Variable weights: {variable_weights}
Reveal sequence: {reveal_sequence}

=== TASK ===
Reason step by step. For each step state:
1. Observation from the data
2. Clinical implication
3. Weight (decisive, important, noted, not_used)

Reconcile the three signal sources:
- LLM reasoning
- ML model scores
- Guideline evidence

Note disagreements explicitly. Then provide your final decision.

{format_instructions}
"""


def _get_llm():
    return ChatOpenAI(
        model=_CFG["llm"]["model"],
        base_url=_CFG["llm"]["base_url"],
        api_key=_CFG["llm"]["api_key"],
        temperature=0.1,
        max_tokens=2048,
    )


def biopsy_decider_agent(state: dict) -> dict:
    features = state["extracted_features"]
    risk = state.get("risk_category", "Unknown")
    data = state["raw_patient_data"]
    mri = state.get("mri_prediction") or {}
    clin = state.get("clinical_prediction") or {}
    evidence = state.get("retrieved_evidence") or []

    parser = PydanticOutputParser(pydantic_object=BiopsyDecisionOutput)

    prompt = BIOPSY_DECIDER_PROMPT.format(
        features=json.dumps(features, indent=2),
        mri_prob=mri.get("csPCa_probability", "N/A"),
        mri_conf=mri.get("confidence", "N/A"),
        clin_prob=clin.get("readiness_score", "N/A"),
        clin_conf=clin.get("confidence", "N/A"),
        clin_contributions=json.dumps(
            clin.get("top_contributing_features", []), indent=2
        ),
        risk_category=risk,
        evidence="\n\n".join(e["content"][:400] for e in evidence[:5]),
        variable_weights=json.dumps(data.get("variable_weights", {})),
        reveal_sequence=json.dumps(data.get("reveal_sequence", [])),
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = _get_llm().invoke(prompt)
        decision = parser.parse(response.content)
    except Exception as e:
        decision = BiopsyDecisionOutput(
            reasoning_chain=[],
            final_decision="no",
            confidence="uncertain",
            key_factors=[],
            summary=f"Parsing error: {str(e)}",
        )

    return {
        "biopsy_decision": decision.model_dump(),
        "audit_log": log_entry(
            state, "biopsy_decider", "made_decision", decision.model_dump()
        ),
    }