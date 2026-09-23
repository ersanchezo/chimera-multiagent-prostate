"""
schemas.py
Pydantic schemas for all agent inputs/outputs and shared data structures.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Extracted Features (Data Extractor output)
# ---------------------------------------------------------------------------

class ExtractedFeatures(BaseModel):
    psa: float
    psav: float
    psap: float
    pirads: int
    psad: float
    vol: float
    dre: str
    bx: str
    age: int
    comorbidity: str
    family_history: str
    psa_trend: List[Dict] = Field(default_factory=list)
    previous_notes_summary: str = ""


# ---------------------------------------------------------------------------
# ML Model Outputs
# ---------------------------------------------------------------------------

class MRIPredictionInput(BaseModel):
    case_id: str
    embedding: List[float] = Field(min_length=64)
    clinical_context: Optional[Dict[str, Any]] = None


class MRIPredictionOutput(BaseModel):
    csPCa_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    threshold_used: float
    model_version: str
    feature_attributions: Optional[List[float]] = None
    error: Optional[str] = None


class ClinicalReadinessInput(BaseModel):
    age: int
    psa: float
    psav: float
    psad: float
    pirads: int
    vol: float
    dre: str
    bx: str
    family_history: str
    comorbidity: str


class ClinicalReadinessOutput(BaseModel):
    readiness_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_contributing_features: List[Dict[str, float]] = Field(default_factory=list)
    model_version: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Risk Stratification
# ---------------------------------------------------------------------------

class RiskStratification(BaseModel):
    nccn_risk_group: str = Field(
        description="Very Low, Low, Intermediate Unfavorable, "
                    "Intermediate Favorable, High, Very High"
    )
    risk_rationale: str
    rule_based_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------------

class DecomposedQueries(BaseModel):
    sub_queries: List[str] = Field(min_length=3, max_length=5)
    retrieval_rationale: str


# ---------------------------------------------------------------------------
# Biopsy Decision
# ---------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    step_number: int
    observation: str
    clinical_implication: str
    weight: Literal["decisive", "important", "noted", "not_used"]


class BiopsyDecisionOutput(BaseModel):
    reasoning_chain: List[ReasoningStep]
    final_decision: Literal["yes", "no"]
    confidence: Literal["clear", "equivocal", "uncertain"]
    key_factors: List[str]
    summary: str = ""


# ---------------------------------------------------------------------------
# Guideline Validation
# ---------------------------------------------------------------------------

class GuidelineValidation(BaseModel):
    is_reasoning_consistent: bool
    consistency_score: float = Field(ge=0.0, le=1.0)
    guideline_consistent: bool
    ml_agreement: bool
    ml_disagreement_notes: Optional[str] = None
    supporting_evidence: List[Dict] = Field(default_factory=list)
    contradicting_evidence: List[Dict] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    suggested_correction: Optional[str] = None
    final_verdict: Literal["VALIDATED", "PARTIALLY_VALIDATED", "REJECTED"]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class FusionResult(BaseModel):
    ensemble_score: float
    final_recommendation: Literal["yes", "no"]
    llm_decision: str
    mri_probability: Optional[float] = None
    clinical_probability: Optional[float] = None
    validator_verdict: str
    disagreements: List[str] = Field(default_factory=list)
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    agent: str
    action: str
    output: Optional[Dict] = None
    timestamp: str