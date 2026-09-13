"""
Model loading, preprocessing, and inference logic for ClaimGuard AI.

The supplied artifact (models/model.pkl) is a scikit-learn
``GridSearchCV`` object wrapping an ``XGBClassifier`` (loaded via
``joblib`` -- the file uses joblib's numpy-array pickling format, not
plain ``pickle``). It expects a 54-column, already-encoded feature
vector as input (see ``utils.schema`` for the full breakdown and the
documented assumptions used to reconstruct the preprocessing step).

This module is defensive by design: every failure mode is caught and
converted into a friendly, non-technical message while full details are
logged internally for operators.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.schema import (
    ClaimInput,
    ENGINEERED_FIELDS,
    FEATURE_NAMES_IN_ORDER,
    INCOME_BIN_EDGES,
    INCOME_BIN_LABELS,
    ONE_HOT_GROUPS,
    REMAINDER_FIELDS,
    VEHICLE_CATEGORY_MAP,
)

logger = logging.getLogger("claimguard_ai")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# Fraud label convention: the training notebook's target column was
# named `fraud_reported`, the universal convention for that column
# across public insurance-fraud datasets is 1 = fraud, 0 = not fraud.
# The pickle's `classes_` attribute confirms the two output classes are
# [0, 1]; we map them accordingly. See README for details.
FRAUD_CLASS_VALUE = 1


class ModelLoadError(Exception):
    """Raised when the model artifact cannot be loaded or is invalid."""


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """
    Load the trained model artifact once per server process.

    Uses joblib (not raw pickle) because the artifact was serialized
    with joblib's numpy-array-aware pickling format.
    """
    if not MODEL_PATH.exists():
        raise ModelLoadError(
            f"Model file not found at expected location: {MODEL_PATH}"
        )

    try:
        import joblib
    except ImportError as exc:
        raise ModelLoadError(
            "The 'joblib' package is required to load this model but is not installed."
        ) from exc

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, converted below
        logger.exception("Failed to load model artifact")
        raise ModelLoadError(
            "The model file could not be loaded. It may be corrupted or "
            "saved with an incompatible library version."
        ) from exc

    if not (hasattr(model, "predict")):
        raise ModelLoadError(
            "The loaded object does not expose a 'predict' method and is "
            "not a usable scikit-learn-compatible model."
        )

    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None and list(expected_cols) != FEATURE_NAMES_IN_ORDER:
        logger.warning(
            "Model feature_names_in_ differs from the schema hard-coded in "
            "utils/schema.py. Predictions may be misaligned."
        )

    return model


def _one_hot(group: str, categories: list, selected: str) -> Dict[str, float]:
    """Build one-hot columns for a single categorical group."""
    return {
        f"ohe__{group}_{cat}": 1.0 if cat == selected else 0.0
        for cat in categories
    }


def _income_bucket(annual_income: float) -> Dict[str, float]:
    """Bucket annual income into the exact quartile bins used at training time."""
    label = INCOME_BIN_LABELS[-1]  # default to top bucket if above all edges
    for i in range(len(INCOME_BIN_EDGES) - 1):
        lo, hi = INCOME_BIN_EDGES[i], INCOME_BIN_EDGES[i + 1]
        if lo < annual_income <= hi:
            label = INCOME_BIN_LABELS[i]
            break
    return {
        f"ohe__income_group_{lbl}": (1.0 if lbl == label else 0.0)
        for lbl in INCOME_BIN_LABELS
    }


def compute_engineered_features(claim: ClaimInput) -> Dict[str, float]:
    """
    Compute default values for the four engineered/derived columns the
    model expects, unless the user has supplied manual overrides.

    Formulas used (documented assumptions -- see utils/schema.py docstring):
      * weekend_claim   = 1 if the claim's day-of-week is Sat/Sun, else 0
                          (unambiguous, derived directly from claim_day_of_week).
      * vehicle_old     = 1 if age_of_vehicle >= 10 years, else 0.
      * claim_ratio     = total_claim / annual_income (0 if income is 0).
      * claim_vehicle_ratio = total_claim / vehicle_price (0 if price is 0).
    """
    if claim.weekend_claim_override is not None:
        weekend_claim = float(bool(claim.weekend_claim_override))
    else:
        weekend_claim = float(claim.claim_day_of_week in ("Saturday", "Sunday"))

    if claim.vehicle_old_override is not None:
        vehicle_old = float(bool(claim.vehicle_old_override))
    else:
        vehicle_old = float(claim.age_of_vehicle >= 10)

    if claim.claim_ratio_override is not None:
        claim_ratio = float(claim.claim_ratio_override)
    else:
        claim_ratio = (
            claim.total_claim / claim.annual_income if claim.annual_income else 0.0
        )

    if claim.claim_vehicle_ratio_override is not None:
        claim_vehicle_ratio = float(claim.claim_vehicle_ratio_override)
    else:
        claim_vehicle_ratio = (
            claim.total_claim / claim.vehicle_price if claim.vehicle_price else 0.0
        )

    return {
        "remainder__claim_ratio": claim_ratio,
        "remainder__vehicle_old": vehicle_old,
        "remainder__weekend_claim": weekend_claim,
        "remainder__claim_vehicle_ratio": claim_vehicle_ratio,
    }


def build_feature_row(claim: ClaimInput) -> pd.DataFrame:
    """Convert a ClaimInput into the exact 54-column DataFrame the model expects."""
    row: Dict[str, float] = {}

    row.update(_one_hot("gender", ONE_HOT_GROUPS["gender"], claim.gender))
    row.update(
        _one_hot("property_status", ONE_HOT_GROUPS["property_status"], claim.property_status)
    )
    row.update(
        _one_hot(
            "claim_day_of_week",
            ONE_HOT_GROUPS["claim_day_of_week"],
            claim.claim_day_of_week,
        )
    )
    row.update(
        _one_hot("accident_site", ONE_HOT_GROUPS["accident_site"], claim.accident_site)
    )
    row.update(_one_hot("channel", ONE_HOT_GROUPS["channel"], claim.channel))
    row.update(
        _one_hot("vehicle_color", ONE_HOT_GROUPS["vehicle_color"], claim.vehicle_color)
    )
    row.update(_income_bucket(claim.annual_income))

    row["ord__vehicle_category"] = float(
        VEHICLE_CATEGORY_MAP.get(claim.vehicle_category, 0)
    )

    remainder_values = {
        "age_of_driver": claim.age_of_driver,
        "marital_status": 1.0 if claim.marital_status == "Married" else 0.0,
        "safety_rating": claim.safety_rating,
        "annual_income": claim.annual_income,
        "high_education": float(claim.high_education),
        "address_change": float(claim.address_change),
        "past_num_of_claims": claim.past_num_of_claims,
        "witness_present": float(claim.witness_present),
        "liab_prct": claim.liab_prct,
        "police_report": float(claim.police_report),
        "age_of_vehicle": claim.age_of_vehicle,
        "vehicle_price": claim.vehicle_price,
        "total_claim": claim.total_claim,
        "injury_claim": claim.injury_claim,
        "policy_deductible": claim.policy_deductible,
        "annual_premium": claim.annual_premium,
        "days_open": claim.days_open,
        "form_defects": claim.form_defects,
        "claim_year": claim.claim_year,
        "claim_month": claim.claim_month,
        "claim_day": claim.claim_day,
    }
    for field_name in REMAINDER_FIELDS:
        row[f"remainder__{field_name}"] = float(remainder_values[field_name])

    row.update(compute_engineered_features(claim))

    df = pd.DataFrame([row])

    # Guarantee exact column presence and ordering expected by the model.
    missing = [c for c in FEATURE_NAMES_IN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Internal error: missing engineered columns: {missing}")

    return df[FEATURE_NAMES_IN_ORDER]


def predict_claim(claim: ClaimInput) -> Dict[str, Any]:
    """
    Run the full prediction pipeline for a single claim.

    Returns a dict with keys:
      - success: bool
      - label: "FRAUDULENT CLAIM" | "VALID CLAIM" (if success)
      - is_fraud: bool
      - confidence: float in [0, 1] or None if unavailable
      - error: user-facing error message (if not success)
    """
    try:
        model = load_model()
    except ModelLoadError as exc:
        logger.error("Model load error: %s", exc)
        return {
            "success": False,
            "error": (
                "The fraud-detection model is currently unavailable. "
                "Please contact support or try again later."
            ),
        }

    try:
        features_df = build_feature_row(claim)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to build feature row from claim input")
        return {
            "success": False,
            "error": (
                "Unable to analyze the claim right now. Please verify the "
                "entered information and try again."
            ),
        }

    try:
        raw_prediction = model.predict(features_df)
        predicted_value = raw_prediction[0]
    except Exception:  # noqa: BLE001
        logger.exception("Model prediction failed")
        return {
            "success": False,
            "error": (
                "Unable to analyze the claim right now. Please verify the "
                "entered information and try again."
            ),
        }

    confidence: Optional[float] = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]
            classes = list(getattr(model, "classes_", range(len(proba))))
            idx = classes.index(predicted_value)
            confidence = float(proba[idx])
    except Exception:  # noqa: BLE001
        logger.warning("predict_proba unavailable or failed; continuing without it")
        confidence = None

    is_fraud = bool(predicted_value == FRAUD_CLASS_VALUE)
    label = "FRAUDULENT CLAIM" if is_fraud else "VALID CLAIM"

    return {
        "success": True,
        "label": label,
        "is_fraud": is_fraud,
        "confidence": confidence,
        "raw_prediction": predicted_value,
    }
