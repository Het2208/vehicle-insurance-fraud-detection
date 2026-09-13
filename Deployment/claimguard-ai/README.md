# ClaimGuard AI

**AI-Powered Insurance Claim Fraud Detection**

A production-styled Streamlit application that wraps a trained XGBoost
classification model to predict whether an insurance claim is
**Fraudulent** or **Valid**, based on policy, customer, incident, and
vehicle details entered through a professional web form.

---

## 1. Project Overview

ClaimGuard AI loads a pre-trained scikit-learn `GridSearchCV`-wrapped
`XGBClassifier` model (`models/model.pkl`) and exposes it through a
polished, single-page Streamlit interface. Users fill in claim details
organized into logical sections, click **Analyze Claim**, and receive
an instant model-driven prediction with a confidence score.

The model itself is **not retrained** by this application — it is used
purely for inference, exactly as supplied.

## 2. Features

- Clean, SaaS-style single-page interface (not a default Streamlit demo)
- Logically grouped claim form: customer, policy, incident, and vehicle sections
- Robust preprocessing that reconstructs the model's exact 54-column
  encoded feature schema from user-friendly form inputs
- Clear **FRAUDULENT CLAIM** / **VALID CLAIM** result banners with
  confidence score and risk level
- Defensive error handling — no raw tracebacks are ever shown to users
- Model is loaded once and cached (`st.cache_resource`) for fast repeated predictions
- Reset button to run another prediction

## 3. Project Structure

```text
claimguard-ai/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml           # Theme & server configuration
├── models/
│   └── model.pkl             # Supplied trained model artifact
└── utils/
    ├── __init__.py
    ├── schema.py              # Feature schema + documented assumptions
    └── model_utils.py         # Model loading, preprocessing, prediction
```

## 4. Installation

### 4.1 Prerequisites

- Python 3.10 or later
- pip

### 4.2 Clone / unzip the project

```bash
cd claimguard-ai
```

### 4.3 Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4.4 Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Running Locally

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## 6. Model Placement

The model file must live at:

```text
models/model.pkl
```

The app locates it using a path relative to `app.py`
(`Path(__file__).resolve().parent / "models" / "model.pkl"`), so no
absolute or OS-specific paths are required. If you retrain or replace
the model, simply overwrite this file — as long as the new model
exposes `.predict()` (and ideally `.predict_proba()`) and expects the
same 54-column feature schema documented in `utils/schema.py`.

## 7. Deployment

The project has no local-only assumptions and uses relative paths
throughout, so it can be deployed to any standard Streamlit-compatible
host (Streamlit Community Cloud, a container platform, or a VM):

1. Push the project (including `models/model.pkl`) to your git repository
   or hosting platform of choice.
2. Set the entry point to `app.py`.
3. Ensure `requirements.txt` is installed during the build step.
4. No environment variables or secrets are required for basic operation.

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| "Model file not found" error on load | `models/model.pkl` missing or renamed | Confirm the file exists at `models/model.pkl` |
| `InconsistentVersionWarning` in logs | scikit-learn version differs slightly from the training environment (1.7.2) | Usually safe to ignore; for strict reproducibility, install `scikit-learn==1.7.2` |
| Model fails to load / corrupted pickle | The `.pkl` was serialized with `joblib.dump()`, not plain `pickle.dump()` — this is expected and already handled internally via `joblib.load()` | No action needed; do not attempt to re-save it with plain `pickle` |
| Prediction always returns the same class | Input values are far outside the training data's distribution | Try more realistic values across all fields |
| App won't start / import errors | Dependencies not installed in the active environment | Re-run `pip install -r requirements.txt` inside the activated virtual environment |

## 9. Model Compatibility Notes

The supplied `.pkl` contains **only the trained classifier**
(`GridSearchCV` wrapping an `XGBClassifier`) — the upstream
preprocessing pipeline (`ColumnTransformer`) used during training was
**not** included in the artifact. `utils/schema.py` reconstructs this
preprocessing step from the model's `feature_names_in_` metadata. Please
review the following before using this app for real decisions:

1. **Verified from the model itself (not guesses):** the one-hot
   categorical groups (gender, home ownership, day of week, accident
   site, sales channel, vehicle color) and the exact income-bracket bin
   edges are read directly from `feature_names_in_`.
2. **Assumption — `vehicle_category` ordinal mapping:** the pickle
   confirms this field was ordinal-encoded, but not which category
   strings map to which codes. The app assumes the three common labels
   `Compact`, `Large`, `Medium`, mapped alphabetically (`Compact=0`,
   `Large=1`, `Medium=2`), matching scikit-learn's default `OrdinalEncoder`
   behavior. If your original training categories differ, update
   `VEHICLE_CATEGORY_MAP` in `utils/schema.py`.
3. **Assumption — engineered ratio features:** `claim_ratio`,
   `vehicle_old`, `weekend_claim`, and `claim_vehicle_ratio` are derived
   metrics not directly collectible from a user. The app computes them
   with documented formulas (see `compute_engineered_features` in
   `utils/model_utils.py`) and lets an advanced user override them
   manually in the "Advanced / Computed Risk Metrics" section of the form.
4. **Fraud label convention:** the model's `classes_` are `[0, 1]`. This
   app assumes the standard convention for the `fraud_reported` target
   used across public insurance-fraud datasets: `1 = fraud`, `0 = not fraud`.

## 10. Production Considerations

- The app never accepts user-uploaded pickle files — the model artifact
  is a trusted, locally bundled file only.
- All internal errors are logged server-side and converted to friendly,
  non-technical messages for end users; no stack traces are ever shown.
- The model is loaded once per server process via `st.cache_resource`,
  keeping repeated predictions fast.
- Before relying on this tool for real claims decisions, validate the
  assumptions in Section 9 against your actual training data and update
  `utils/schema.py` accordingly.
- Predictions are probabilistic model outputs, not certainties, and
  should be reviewed by a qualified claims professional according to
  your organization's policies.
