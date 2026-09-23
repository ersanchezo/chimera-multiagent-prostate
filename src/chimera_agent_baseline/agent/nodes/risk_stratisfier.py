"""
risk_stratifier.py
Agent 2: Risk Stratifier — LLM + rule-based NCCN classification.
"""

import json
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from schemas.schemas import RiskStratification
from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

RISK_PROMPT = """You are a urology risk stratification agent.
Given the following extracted features, classify this patient into an NCCN risk group.

Features:
{features}

NCCN risk groups:
- Very Low: PSA <10, Grade Group 1, <3 cores positive, ≤50% core involvement
- Low: PSA <10, Grade Group 1
- Intermediate Unfavorable: Grade Group 2-3, or PSA 10-20
- Intermediate Favorable: Grade Group 2, PSA 10-20, <50% cores positive
- High: Grade Group 4-5, or PSA >20
- Very High: T3b-T4

{format_instructions}
"""


def _get_llm():
    return ChatOpenAI(
        model=_CFG["llm"]["model"],
        base_url=_CFG["llm"]["base_url"],
        api_key=_CFG["llm"]["api_key"],
        temperature=0.0,
        max_tokens=1024,
    )


def risk_stratifier_agent(state: dict) -> dict:
    features = state["extracted_features"]
    parser = PydanticOutputParser(pydantic_object=RiskStratification)

    prompt = RISK_PROMPT.format(
        features=json.dumps(features, indent=2),
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = _get_llm().invoke(prompt)
        risk = parser.parse(response.content)
    except Exception as e:
        risk = RiskStratification(
            nccn_risk_group="Unknown",
            risk_rationale=f"Parsing error: {str(e)}",
            rule_based_score=None,
        )

    return {
        "risk_category": risk.nccn_risk_group,
        "audit_log": log_entry(
            state, "risk_stratifier", "classified_risk", risk.model_dump()
        ),
    }