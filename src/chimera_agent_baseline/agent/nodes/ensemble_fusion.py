"""
ensemble_fusion.py
Agent 7: Ensemble Fusion — combines LLM decision, ML scores, and validator verdict.
Rule-based (no LLM) for reproducibility and auditability.
"""

import yaml

from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

_W = _CFG["fusion"]["weights"]
_UNCERTAIN_BAND = _CFG["fusion"]["uncertain_band"]
_DISAGREE_THRESH = _CFG["fusion"]["disagreement_threshold"]
_MIN_DISAGREE = _CFG["fusion"]["min_disagreements_for_review"]
_HUMAN_THRESH = _CFG["fusion"]["human_review_threshold"]


def ensemble_fusion_node(state: dict) -> dict:
    """
    Combine LLM decision, ML scores, and validator verdict.
    Returns final recommendation plus agreement flags.
    """
    decision = state.get("biopsy_decision") or {}
    mri = state.get("mri_prediction") or {}
    clin = state.get("clinical_prediction") or {}
    validation = state.get("validation_result") or {}

    llm_yes = decision.get("final_decision", "").lower() == "yes"
    llm_conf = {"clear": 1.0, "equivocal": 0.6, "uncertain": 0.3}.get(
        decision.get("confidence", "uncertain"), 0.3
    )

    mri_prob = mri.get("csPCa_probability")
    mri_w = mri.get("confidence", 0.0) or 0.0
    clin_prob = clin.get("readiness_score")
    clin_w = clin.get("confidence", 0.0) or 0.0

    verdict = validation.get("final_verdict", "REJECTED")
    val_score = {"VALIDATED": 1.0, "PARTIALLY_VALIDATED": 0.6, "REJECTED": 0.0}.get(
        verdict, 0.0
    )

    signals = [(1.0 if llm_yes else 0.0, _W["llm_decision"] * llm_conf)]
    if mri_prob is not None:
        signals.append((mri_prob, _W["mri_model"] * mri_w))
    if clin_prob is not None:
        signals.append((clin_prob, _W["clinical_model"] * clin_w))
    signals.append((val_score, _W["validator"]))

    total_w = sum(w for _, w in signals)
    ensemble_score = (
        sum(v * w for v, w in signals) / total_w if total_w else 0.0
    )

    # Agreement flags
    disagreements = []
    llm_bin = 1.0 if llm_yes else 0.0
    if mri_prob is not None and abs(llm_bin - mri_prob) > _DISAGREE_THRESH:
        disagreements.append("LLM vs MRI model")
    if clin_prob is not None and abs(llm_bin - clin_prob) > _DISAGREE_THRESH:
        disagreements.append("LLM vs clinical model")
    if verdict == "REJECTED":
        disagreements.append("Validator rejected reasoning")

    final_recommendation = "yes" if ensemble_score >= 0.5 else "no"

    requires_review = (
        len(disagreements) >= _MIN_DISAGREE
        or ensemble_score < _HUMAN_THRESH
        or (_UNCERTAIN_BAND[0] <= ensemble_score <= _UNCERTAIN_BAND[1])
    )

    fusion = {
        "ensemble_score": round(ensemble_score, 4),
        "final_recommendation": final_recommendation,
        "llm_decision": "yes" if llm_yes else "no",
        "mri_probability": mri_prob,
        "clinical_probability": clin_prob,
        "validator_verdict": verdict,
        "disagreements": disagreements,
        "requires_human_review": requires_review,
    }

    return {
        "fusion_result": fusion,
        "audit_log": log_entry(
            state, "fusion", "fused_signals", fusion
        ),
    }