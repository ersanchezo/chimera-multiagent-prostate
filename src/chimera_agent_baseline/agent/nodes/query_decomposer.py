"""
query_decomposer.py
Agent 3: Query Decomposer — breaks the case into 3-5 retrieval sub-queries.
"""

import json
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from schemas.schemas import DecomposedQueries
from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

DECOMPOSER_PROMPT = """You are a clinical query decomposition agent.
Break down a prostate cancer biopsy decision case into 3-5 focused sub-queries
that can each retrieve relevant clinical guideline sections.

=== CASE SUMMARY ===
Age: {age}
PSA: {psa} ng/mL
PSA velocity: {psav} ng/mL/yr
PI-RADS: {pirads}
PSAD: {psad} ng/mL/mL
DRE: {dre}
Previous biopsy: {bx}
Risk category: {risk_category}
MRI model csPCa probability: {mri_prob}
Clinical readiness score: {clin_prob}

=== AVAILABLE GUIDELINE TOPICS ===
- PSA thresholds and velocity criteria for biopsy
- PI-RADS scoring and management recommendations
- DRE findings and biopsy indication
- PSAD thresholds and clinical significance
- Prior positive biopsy and active surveillance criteria
- NCCN risk stratification and biopsy recommendations

=== TASK ===
Decompose this case into 3-5 sub-queries. Each sub-query should:
1. Target ONE specific clinical criterion
2. Include the relevant patient value(s)
3. Be phrased as a natural language question
4. Be independently searchable

{format_instructions}
"""


def _get_llm():
    return ChatOpenAI(
        model=_CFG["llm"]["model"],
        base_url=_CFG["llm"]["base_url"],
        api_key=_CFG["llm"]["api_key"],
        temperature=0.1,
        max_tokens=1024,
    )


def query_decomposer_agent(state: dict) -> dict:
    features = state["extracted_features"]
    risk = state.get("risk_category", "Unknown")
    mri = state.get("mri_prediction") or {}
    clin = state.get("clinical_prediction") or {}

    parser = PydanticOutputParser(pydantic_object=DecomposedQueries)

    prompt = DECOMPOSER_PROMPT.format(
        age=features.get("age"),
        psa=features.get("psa"),
        psav=features.get("psav"),
        pirads=features.get("pirads"),
        psad=features.get("psad"),
        dre=features.get("dre"),
        bx=features.get("bx"),
        risk_category=risk,
        mri_prob=mri.get("csPCa_probability", "N/A"),
        clin_prob=clin.get("readiness_score", "N/A"),
        format_instructions=parser.get_format_instructions(),
    )

    try:
        response = _get_llm().invoke(prompt)
        decomposed = parser.parse(response.content)
    except Exception as e:
        decomposed = DecomposedQueries(
            sub_queries=[
                f"What is the biopsy recommendation for PI-RADS {features.get('pirads')}?",
                f"What PSA threshold warrants biopsy for a {features.get('age')}-year-old?",
                f"What does a DRE finding of '{features.get('dre')}' indicate?",
                f"What PSAD value is clinically significant? PSAD is {features.get('psad')}",
            ],
            retrieval_rationale=f"Fallback template: {str(e)}",
        )

    return {
        "decomposed_queries": decomposed.model_dump(),
        "audit_log": log_entry(
            state, "query_decomposer", "decomposed_query",
            decomposed.model_dump(),
        ),
    }