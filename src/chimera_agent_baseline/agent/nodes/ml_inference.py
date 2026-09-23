"""
ml_inference.py
Agent 1b: ML Inference — deterministic execution of both ML models in parallel.
"""

import concurrent.futures
import yaml

from tools.ml_tools import (
    predict_cspca_from_mri_embedding,
    score_clinical_biopsy_readiness,
)
from utils.audit import log_entry


with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

_TIMEOUT = _CFG["ml_models"]["timeout_seconds"]


def ml_inference_node(state: dict) -> dict:
    """
    Run both ML models in parallel. Their outputs are stored in state
    for downstream agents (biopsy_decider, validator, fusion).
    """
    features = state["extracted_features"]
    patient_data = state["raw_patient_data"]
    embeddings = patient_data.get("mri_embedding")

    mri_result = None
    clinical_result = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        mri_future = None
        if embeddings:
            mri_future = ex.submit(
                predict_cspca_from_mri_embedding.func,
                case_id=state["case_id"],
                embedding=embeddings,
                clinical_context={
                    "psa": features["psa"],
                    "psad": features["psad"],
                    "pirads": features["pirads"],
                    "vol": features["vol"],
                },
            )

        clin_future = ex.submit(
            score_clinical_biopsy_readiness.func,
            age=features["age"],
            psa=features["psa"],
            psav=features["psav"],
            psad=features["psad"],
            pirads=features["pirads"],
            vol=features["vol"],
            dre=features["dre"],
            bx=features["bx"],
            family_history=features["family_history"],
            comorbidity=features["comorbidity"],
        )

        if mri_future:
            try:
                mri_result = mri_future.result(timeout=_TIMEOUT)
            except Exception as e:
                mri_result = {
                    "error": str(e),
                    "csPCa_probability": None,
                    "confidence": 0.0,
                }

        try:
            clinical_result = clin_future.result(timeout=_TIMEOUT)
        except Exception as e:
            clinical_result = {
                "error": str(e),
                "readiness_score": None,
                "confidence": 0.0,
            }

    return {
        "mri_prediction": mri_result,
        "clinical_prediction": clinical_result,
        "audit_log": log_entry(
            state, "ml_inference", "ran_ml_models",
            {"mri": mri_result, "clinical": clinical_result},
        ),
    }