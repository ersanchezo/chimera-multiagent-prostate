"""
ml_models.py
Model loading and inference logic for both ML tools.
Kept separate from LangChain wrappers so models can be tested in isolation.
"""

import joblib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


class MRIClassifier:
    """
    MRI embedding classifier for csPCa probability.
    Supports sklearn-style models (joblib) or PyTorch modules.
    """

    def __init__(self, model_path: str, embedding_dim: int = 1024):
        self.model_path = Path(model_path)
        self.embedding_dim = embedding_dim
        self.version = f"mri-emb-{self.model_path.stem}"

        if not self.model_path.exists():
            raise FileNotFoundError(f"MRI model not found: {model_path}")

        if self.model_path.suffix in (".pt", ".pth"):
            self.model = torch.load(self.model_path, map_location="cpu")
            self.model.eval()
            self.is_torch = True
        else:
            self.model = joblib.load(self.model_path)
            self.is_torch = False

    def predict(
        self,
        embedding: List[float],
        clinical_context: Optional[Dict] = None,
    ) -> Dict:
        """Return csPCa probability and confidence."""
        X = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        if X.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {X.shape[1]}"
            )

        if clinical_context:
            ctx = np.array([
                clinical_context.get("psa", 0),
                clinical_context.get("psad", 0),
                clinical_context.get("pirads", 0),
                clinical_context.get("vol", 0),
            ], dtype=np.float32).reshape(1, -1)
            X = np.hstack([X, ctx])

        if self.is_torch:
            with torch.no_grad():
                tensor = torch.from_numpy(X)
                prob = float(torch.sigmoid(self.model(tensor)).item())
        else:
            prob = float(self.model.predict_proba(X)[0, 1])

        # Confidence as distance from decision boundary
        conf = float(min(abs(prob - 0.5) * 2, 1.0))

        return {
            "csPCa_probability": prob,
            "confidence": conf,
            "threshold_used": 0.5,
            "model_version": self.version,
        }


class ClinicalReadinessClassifier:
    """
    Clinical readiness classifier — probability that the clinical picture
    warrants biopsy.
    """

    FEATURE_NAMES = [
        "age", "psa", "psav", "psad", "pirads", "vol",
        "dre", "bx", "family_history", "comorbidity",
    ]

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.version = f"clinical-{self.model_path.stem}"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Clinical model not found: {model_path}")

        self.model = joblib.load(self.model_path)

    @staticmethod
    def _encode_dre(dre: str) -> int:
        return 1 if dre.lower() in ("nodus", "nodule", "abnormal") else 0

    @staticmethod
    def _encode_bx(bx: str) -> int:
        return 1 if bx.lower() in ("positive", "yes") else 0

    @staticmethod
    def _encode_fh(fh: str) -> int:
        return 1 if fh.lower() == "yes" else 0

    @staticmethod
    def _encode_comorbidity(com: str) -> int:
        return 1 if "obesity" in com.lower() else 0

    def predict(
        self,
        age: int, psa: float, psav: float, psad: float,
        pirads: int, vol: float, dre: str, bx: str,
        family_history: str, comorbidity: str,
    ) -> Dict:
        """Return readiness score and confidence."""
        features = np.array([[
            age, psa, psav, psad, pirads, vol,
            self._encode_dre(dre),
            self._encode_bx(bx),
            self._encode_fh(family_history),
            self._encode_comorbidity(comorbidity),
        ]], dtype=np.float32)

        score = float(self.model.predict_proba(features)[0, 1])
        conf = float(min(abs(score - 0.5) * 2, 1.0))

        contributions = []
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for name, imp, val in zip(self.FEATURE_NAMES, importances, features[0]):
                contributions.append({
                    "feature": name,
                    "importance": float(imp),
                    "value": float(val),
                })
            contributions.sort(key=lambda x: x["importance"], reverse=True)
            contributions = contributions[:5]

        return {
            "readiness_score": score,
            "confidence": conf,
            "top_contributing_features": contributions,
            "model_version": self.version,
        }