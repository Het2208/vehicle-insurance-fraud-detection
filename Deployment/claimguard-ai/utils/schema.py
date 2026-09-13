"""
Feature schema for the ClaimGuard AI fraud-detection model.

This module encodes the EXACT 54-column input schema that the trained
model (a GridSearchCV-wrapped XGBClassifier) expects, as read directly
from the pickled estimator's ``feature_names_in_`` attribute:

    ['ohe__gender_F', 'ohe__gender_M', 'ohe__property_status_Own', ...]

The column names reveal how the original training pipeline was built:

* ``ohe__<field>_<category>``   -> one-hot encoded categorical column
                                    (all categories were kept, i.e. the
                                    encoder was NOT configured to drop
                                    the first level).
* ``ord__<field>``              -> ordinal-encoded categorical column.
* ``remainder__<field>``        -> numeric column passed through
                                    unchanged by the ColumnTransformer.

Only a trained classifier was supplied in the .pkl file -- the upstream
ColumnTransformer / preprocessing pipeline itself was NOT included in
the artifact. This module reconstructs the preprocessing step needed to
go from human-entered claim details to the exact 54-column, correctly
ordered feature vector the model expects.

IMPORTANT COMPATIBILITY NOTES (please read):

1. One-hot groups (gender, property_status, claim_day_of_week,
   accident_site, channel, vehicle_color, income_group) and the exact
   income_group bin edges are read verbatim from the model's
   ``feature_names_in_`` metadata, so these are NOT guesses.

2. ``ord__vehicle_category`` - the pickle only tells us the encoded
   field is named "vehicle_category"; it does NOT tell us which string
   values map to which ordinal codes. Scikit-learn's OrdinalEncoder
   assigns codes in sorted (alphabetical) order of the categories seen
   during training. Based on the surrounding feature set (a well-known
   public auto-insurance claims schema), the three most common category
   labels for this field are "Compact", "Medium", and "Large". We map
   them alphabetically: Compact=0, Large=1, Medium=2. If your original
   training data used different category labels, update
   ``VEHICLE_CATEGORY_MAP`` below to match.

3. Four engineered numeric columns are not directly collectible from a
   user because they are *derived* metrics: ``claim_ratio``,
   ``vehicle_old``, ``weekend_claim``, ``claim_vehicle_ratio``. The app
   auto-computes sensible defaults for these (documented in
   ``compute_engineered_features``) and lets an advanced user override
   them manually before running a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# One-hot encoded categorical groups
# (extracted verbatim from the model's feature_names_in_)
# ---------------------------------------------------------------------------

ONE_HOT_GROUPS: Dict[str, List[str]] = {
    "gender": ["F", "M"],
    "property_status": ["Own", "Rent"],
    "claim_day_of_week": [
        "Friday",
        "Monday",
        "Saturday",
        "Sunday",
        "Thursday",
        "Tuesday",
        "Wednesday",
    ],
    "accident_site": ["Highway", "Local", "Parking Lot"],
    "channel": ["Broker", "Online", "Phone"],
    "vehicle_color": [
        "black",
        "blue",
        "gray",
        "other",
        "red",
        "silver",
        "white",
    ],
}

# Income bracket bin edges, read verbatim from the trained model's
# one-hot column names for `income_group`.
INCOME_BIN_EDGES: List[float] = [-1.601, 56899.2, 60899.2, 64699.2, 257313.6]
INCOME_BIN_LABELS: List[str] = [
    "(-1.601, 56899.2)",
    "(56899.2, 60899.2)",
    "(60899.2, 64699.2)",
    "(64699.2, 257313.6)",
]

# ---------------------------------------------------------------------------
# Ordinal encoded field
# ---------------------------------------------------------------------------

# ASSUMPTION (see module docstring, note 2): alphabetical ordinal mapping.
VEHICLE_CATEGORY_MAP: Dict[str, int] = {"Compact": 0, "Large": 1, "Medium": 2}

# ---------------------------------------------------------------------------
# Passthrough ("remainder") numeric fields, in the exact order the model
# expects them (order matters only within the overall final vector, which
# `build_feature_row` assembles using FEATURE_NAMES_IN_ANYWAY, so this list
# is purely for building the UI form).
# ---------------------------------------------------------------------------

REMAINDER_FIELDS: List[str] = [
    "age_of_driver",
    "marital_status",
    "safety_rating",
    "annual_income",
    "high_education",
    "address_change",
    "past_num_of_claims",
    "witness_present",
    "liab_prct",
    "police_report",
    "age_of_vehicle",
    "vehicle_price",
    "total_claim",
    "injury_claim",
    "policy_deductible",
    "annual_premium",
    "days_open",
    "form_defects",
    "claim_year",
    "claim_month",
    "claim_day",
]

# Engineered / derived numeric fields (see module docstring, note 3).
ENGINEERED_FIELDS: List[str] = [
    "claim_ratio",
    "vehicle_old",
    "weekend_claim",
    "claim_vehicle_ratio",
]

# The exact, full ordered list of 54 columns the model expects.
# This MUST match model.feature_names_in_ / model.get_booster? no -
# it is validated at runtime against the loaded model in model_utils.py.
FEATURE_NAMES_IN_ORDER: List[str] = (
    [f"ohe__gender_{c}" for c in ONE_HOT_GROUPS["gender"]]
    + [f"ohe__property_status_{c}" for c in ONE_HOT_GROUPS["property_status"]]
    + [f"ohe__claim_day_of_week_{c}" for c in ONE_HOT_GROUPS["claim_day_of_week"]]
    + [f"ohe__accident_site_{c}" for c in ONE_HOT_GROUPS["accident_site"]]
    + [f"ohe__channel_{c}" for c in ONE_HOT_GROUPS["channel"]]
    + [f"ohe__vehicle_color_{c}" for c in ONE_HOT_GROUPS["vehicle_color"]]
    + [f"ohe__income_group_{c}" for c in INCOME_BIN_LABELS]
    + ["ord__vehicle_category"]
    + [f"remainder__{c}" for c in REMAINDER_FIELDS]
    + [f"remainder__{c}" for c in ENGINEERED_FIELDS]
)


@dataclass(frozen=True)
class ClaimInput:
    """Structured, human-friendly claim data collected from the UI form."""

    # Customer / policy
    gender: str = "F"
    marital_status: str = "Single"
    age_of_driver: int = 35
    high_education: bool = False
    annual_income: float = 50000.0
    property_status: str = "Own"
    address_change: bool = False
    safety_rating: int = 80

    # Policy / claim details
    channel: str = "Online"
    policy_deductible: float = 500.0
    annual_premium: float = 1200.0
    liab_prct: int = 50
    police_report: bool = False
    witness_present: bool = False
    past_num_of_claims: int = 0
    form_defects: int = 0
    days_open: int = 10

    # Incident information
    claim_year: int = 2024
    claim_month: int = 1
    claim_day: int = 1
    claim_day_of_week: str = "Monday"
    accident_site: str = "Local"

    # Vehicle information
    vehicle_category: str = "Compact"
    vehicle_color: str = "white"
    vehicle_price: float = 20000.0
    age_of_vehicle: int = 5

    # Claim amounts
    total_claim: float = 5000.0
    injury_claim: float = 0.0

    # Engineered / derived (auto-computed, user-overridable)
    claim_ratio_override: Optional[float] = None
    vehicle_old_override: Optional[bool] = None
    weekend_claim_override: Optional[bool] = None
    claim_vehicle_ratio_override: Optional[float] = None
