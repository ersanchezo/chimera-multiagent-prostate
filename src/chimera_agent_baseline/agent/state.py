"""
state.py
Shared LangGraph state definition.
"""

from typing import Any, Dict, List, Optional


class MultiAgentState(dict):
    """
    State passed between agents in the LangGraph.
    Using a dict subclass so LangGraph can merge partial updates.
    """
    case_id: str
    raw_patient_data: Dict[str, Any]
    extracted_features: Optional[Dict]
    mri_prediction: Optional[Dict]
    clinical_prediction: Optional[Dict]
    risk_category: Optional[str]
    decomposed_queries: Optional[Dict]
    retrieved_evidence: Optional[List[Dict]]
    biopsy_decision: Optional[Dict]
    validation_result: Optional[Dict]
    decision_accepted: Optional[bool]
    fusion_result: Optional[Dict]
    explanation: Optional[str]
    audit_log: List[Dict]
    error: Optional[str]


def initial_state(case_id: str, patient_data: Dict[str, Any]) -> MultiAgentState:
    """Factory for a fresh state object."""
    return {
        "case_id": case_id,
        "raw_patient_data": patient_data,
        "extracted_features": None,
        "mri_prediction": None,
        "clinical_prediction": None,
        "risk_category": None,
        "decomposed_queries": None,
        "retrieved_evidence": None,
        "biopsy_decision": None,
        "validation_result": None,
        "decision_accepted": None,
        "fusion_result": None,
        "explanation": None,
        "audit_log": [],
        "error": None,
    }