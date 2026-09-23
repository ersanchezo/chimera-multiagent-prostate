"""
ml_tools.py
LangChain tool wrappers around the ML models.
These are the callable interfaces used by the ML inference node.
"""

from langchain_core.tools import tool
from schemas.schemas import (
    MRIPredictionInput,
    ClinicalReadinessInput,
)
from tools.ml_models import MRIClassifier, ClinicalReadinessClassifier
import yaml


# Load config
with open("config/config.yaml") as f:
    _CFG = yaml.safe_load(f)

# Singleton model instances
_MRI_MODEL = None
_CLINICAL_MODEL = None


def _get_mri_model() -> MRIClassifier:
    global _MRI_MODEL
    if _MRI_MODEL is None:
        _MRI_MODEL = MRIClassifier(
            _CFG["ml_models"]["mri_classifier_path"],
            _CFG["ml_models"]["mri_embedding_dim"],
        )
    return _MRI_MODEL


def _get_clinical_model() -> ClinicalReadinessClassifier:
    global _CLINICAL_MODEL
    if _CLINICAL_MODEL is None:
        _CLINICAL_MODEL = ClinicalReadinessClassifier(
            _CFG["ml_models"]["clinical_classifier_path"]
        )
    return _CLINICAL_MODEL


@tool("predict_cspca_from_mri_embedding", args_schema=MRIPredictionInput)
def predict_cspca_from_mri_embedding(
    case_id: str,
    embedding: list[float],
    clinical_context: dict | None = None,
) -> dict:
    """
    Run the MRI embedding classifier to predict clinically significant
    prostate cancer (csPCa) probability. Use when MRI embeddings are
    available and a model-based estimate is needed to complement PI-RADS.
    """
    try:
        model = _get_mri_model()
        return model.predict(embedding, clinical_context)
    except Exception as e:
        return {
            "error": str(e),
            "csPCa_probability": None,
            "confidence": 0.0,
            "model_version": "mri-emb-unknown",
        }


@tool("score_clinical_biopsy_readiness", args_schema=ClinicalReadinessInput)
def score_clinical_biopsy_readiness(
    age: int, psa: float, psav: float, psad: float,
    pirads: int, vol: float, dre: str, bx: str,
    family_history: str, comorbidity: str,
) -> dict:
    """
    Score the clinical picture for biopsy readiness using a trained
    classifier. Returns a probability that the case warrants biopsy.
    """
    try:
        model = _get_clinical_model()
        return model.predict(
            age, psa, psav, psad, pirads, vol,
            dre, bx, family_history, comorbidity,
        )
    except Exception as e:
        return {
            "error": str(e),
            "readiness_score": None,
            "confidence": 0.0,
            "model_version": "clinical-readiness-unknown",
        }


ML_TOOLS = [predict_cspca_from_mri_embedding, score_clinical_biopsy_readiness]