"""
DSC 355 – Week 11 / Milestone 5: Final Project Model Deployment
App: Teen Depression Risk Prediction (NSCH 2021)
Author: Hali Sarah Parsons
Date: November 2026
Modified By: Claude (2026-11-10: fixed input encoding and input scales, see CHANGES below)

PURPOSE
-------
This Streamlit app deploys the final-project classification model trained on
the National Survey of Children's Health (NSCH) 2021 dataset, filtered to
adolescents aged 12–17. The model predicts the binary target:

    K2Q33A = 1  → teen has been diagnosed with depression
    K2Q33A = 0  → teen has not been diagnosed with depression

The goal is to surface the model in an accessible interface so that a
clinician, school counselor, or researcher could enter a teen's characteristics
and receive a risk assessment—useful as one signal in a broader screening
process (never as a sole diagnostic tool).

CHANGES (2026-11-10)
--------------------
1. Prediction-time encoding: training label-encodes the float codes ("1.0", "2.0"),
   but the old form looked up int-style strings ("1", "2"). Every lookup failed and
   silently fell back to 0, so the categorical answers (anxiety diagnosis, emotional
   items, sex, race, ...) never changed the prediction. Inputs are now looked up the
   same way they were encoded in training.
2. Input scales: the form offered hours / days / 1-4 scales, but the NSCH file stores
   coded values (e.g. SCREENTIME 1-5, FPL_I1 50-400). The form now offers exactly the
   codes found in the data, with defaults at the most common value.

HOW TO RUN
----------
  pip install streamlit scikit-learn pandas
  streamlit run app_week11.py

The app expects nsch_2021_teens.csv in the same directory, OR it will look
one level up. Adjust DATA_PATH below if needed.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os

# ── Third-party libraries ─────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Teen Depression Risk – DSC 355 Final Project",
    page_icon="🧠",
    layout="wide"
)

# =============================================================================
# SECTION 1: PATH CONFIGURATION
# =============================================================================
# nsch_2021_teens.csv lives with the project files; check the script directory
# first, then one level up (e.g., if running from a subfolder).
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "nsch_2021_teens.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(_HERE, "..", "nsch_2021_teens.csv")

# =============================================================================
# SECTION 2: FEATURE DEFINITIONS
# =============================================================================
# These 18 columns were selected after EDA in Milestones 1–2.  They cover
# four domains: lifestyle (sleep, screen time, activity), demographics
# (age, sex, race), mental/physical health co-morbidities, and social factors
# (bullying, poverty, language spoken at home).
# =============================================================================

TARGET = "K2Q33A"           # Binary: 1 = depression diagnosis, 0 = no

# Features kept after EDA and feature-importance analysis
FEATURE_COLS = [
    "SCREENTIME",           # Average daily screen time (hours)
    "HOURSLEEP",            # Average hours of sleep per night
    "PHYSACTIV",            # Days of physical activity in past week
    "BEDTIME",              # Consistent bedtime (1=yes)
    "K10Q11",               # Feels anxious / nervous
    "K10Q12",               # Worries a lot
    "K10Q13",               # Feels sad or unhappy
    "K10Q14",               # Feels angry
    "SC_AGE_YEARS",         # Teen's age in years (12–17)
    "SC_SEX",               # Sex (1=male, 2=female)
    "SC_RACE_R",            # Race/ethnicity category
    "SC_HISPANIC_R",        # Hispanic origin
    "K2Q31A",               # ADHD diagnosis
    "K2Q32A",               # Anxiety diagnosis
    "BULLIED_R",            # Bullying victimization
    "FPL_I1",               # Federal poverty level category
    "METRO_YN",             # Metropolitan area (1=yes)
    "HHLANGUAGE",           # Household language (1=English)
]

# Columns that need LabelEncoding (they may come in as codes but could be
# object dtype depending on how the CSV was saved)
CATEGORICAL_COLS = [
    "BEDTIME", "SC_SEX", "SC_RACE_R", "SC_HISPANIC_R",
    "K2Q31A", "K2Q32A", "BULLIED_R", "METRO_YN", "HHLANGUAGE",
    "K10Q11", "K10Q12", "K10Q13", "K10Q14",
]


# =============================================================================
# SECTION 3: DATA LOADING & PREPROCESSING
# =============================================================================

@st.cache_data
def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load nsch_2021_teens.csv, keep only the columns we need, drop rows
    with a missing target, and coerce types.
    """
    df = pd.read_csv(path, low_memory=False)

    # Keep only the columns we need (target + features)
    all_cols = [TARGET] + FEATURE_COLS
    df = df[[c for c in all_cols if c in df.columns]].copy()

    # Drop rows where the target is missing (can't train or evaluate on those)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # In NSCH: K2Q33A = 1 → depression diagnosis, K2Q33A = 2 → no diagnosis
    # Recode to standard binary: 1 → diagnosed (positive class), 0 → not diagnosed
    df[TARGET] = df[TARGET].map({1: 1, 2: 0}).fillna(0).astype(int)

    # For survey data, 90–99 often means "refused" or "not applicable"
    # Treat these as NaN and fill with the column median/mode
    for col in df.columns:
        if col == TARGET:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            # Recode high sentinel values only for Likert-scale columns
            if col in ["K10Q11","K10Q12","K10Q13","K10Q14","BULLIED_R"]:
                df[col] = df[col].replace({90: np.nan, 95: np.nan, 99: np.nan})
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================
# We use RandomForestClassifier for several reasons:
#   • Handles mixed numeric / ordinal features without requiring scaling
#   • Naturally models non-linear relationships and feature interactions
#     (important here: anxiety + poor sleep may interact to increase risk)
#   • Provides feature importances that are interpretable to non-data people
#   • Reported best in the Week 7 evaluation. Re-checked on the final 18 features,
#     logistic regression scores about the same (ROC-AUC ~0.85), so the forest is
#     kept mainly for its interactions and built-in importances
# =============================================================================

@st.cache_resource
def train_model(data_path: str):
    """
    Train a RandomForestClassifier on the NSCH teen dataset.

    Returns model, feature metadata, encoders, and evaluation metrics.
    """
    df = load_and_clean(data_path)

    # Encode any object/bool columns that ended up categorical
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    # Only keep features that actually exist in this CSV
    available_features = [c for c in FEATURE_COLS if c in df_enc.columns]
    X = df_enc[available_features]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # RandomForest with class_weight='balanced' to handle the imbalance
    # (depression diagnosis is ~22% of teens in NSCH – the minority class
    # deserves equal treatment so we don't just predict "no depression" always)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",    # Compensate for class imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred, target_names=["No Diagnosis","Diagnosis"])
    cm      = confusion_matrix(y_test, y_pred)

    return clf, encoders, available_features, acc, auc, report, cm


# =============================================================================
# SECTION 5: APP LAYOUT – HEADER
# =============================================================================

st.title("🧠 Teen Depression Risk Prediction")
st.markdown(
    """
    **DSC 355 – Milestone 5 | Final Project Deployment**
    *Hali Sarah Parsons | November 2026*

    ---
    This application uses a **Random Forest classifier** trained on the
    **2021 National Survey of Children's Health (NSCH)** to estimate the
    likelihood that a teenager (ages 12–17) has received a depression diagnosis.

    > ⚠️ **This tool is for educational and research purposes only.**
    > It is not a clinical diagnostic instrument and should never replace
    > evaluation by a qualified mental health professional.
    """
)

# =============================================================================
# SECTION 6: LOAD DATA & TRAIN
# =============================================================================

if not os.path.exists(DATA_PATH):
    st.error(
        f"❌ Dataset not found at: `{DATA_PATH}`\n\n"
        "Place `nsch_2021_teens.csv` in the same folder as this script."
    )
    st.stop()

with st.spinner("Loading NSCH data and training model …"):
    clf, encoders, available_features, acc, auc, report, cm = train_model(DATA_PATH)

# =============================================================================
# SECTION 7: SIDEBAR – MODEL METADATA & PERFORMANCE
# =============================================================================

with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Test Accuracy", f"{acc:.1%}")
    st.metric("ROC-AUC", f"{auc:.3f}")
    st.markdown("---")
    st.markdown(
        """
        - **Algorithm:** Random Forest
        - **Dataset:** NSCH 2021 (teens 12–17)
        - **Records:** 15,787
        - **Target:** K2Q33A (depression dx)
        - **Train/Test:** 80/20 split
        - **Class balancing:** `class_weight='balanced'`
        """
    )
    with st.expander("Classification Report"):
        st.code(report)
    with st.expander("Confusion Matrix"):
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Dx", "Actual: Dx"],
            columns=["Predicted: No Dx", "Predicted: Dx"]
        )
        st.dataframe(cm_df)

# =============================================================================
# SECTION 8: INPUT FORM
# =============================================================================
# Why the options come from the data: the NSCH file stores coded values (for example
# SCREENTIME is 1-5, not hours; FPL_I1 is 50-400). The model only ever saw those
# codes, so the form must offer exactly those codes or the model is asked about
# values it has never seen. Each default is the most common code in the data, so
# the form opens as a "typical teen" instead of whatever option happens to be first.
# =============================================================================

clean_df = load_and_clean(DATA_PATH)


def code_select(label, col, names=None):
    """
    Dropdown limited to the codes that exist in the training data for `col`.
    `names` optionally maps a code to readable text; codes without a name are
    shown as the raw NSCH code so nothing is mislabeled.
    """
    codes = sorted(clean_df[col].dropna().unique().tolist())
    default = clean_df[col].mode()[0]
    fmt = (lambda c: f"{names.get(int(c), int(c))}") if names else (lambda c: f"{int(c)}")
    return st.selectbox(label, codes, index=codes.index(default), format_func=fmt)


YES_NO = {1: "Yes", 2: "No"}

st.header("Enter Teen Characteristics")
st.markdown("Adjust the sliders and dropdowns to match the teen's profile, then click **Assess Risk**. "
            "Where a question has no plain-language label, the dropdown shows the NSCH survey code.")

left, right = st.columns(2)

with left:
    st.subheader("Lifestyle & Behavior")
    SCREENTIME = code_select("Weekday screen time (NSCH code, higher = more)", "SCREENTIME")
    HOURSLEEP  = code_select("Nightly sleep (NSCH code)", "HOURSLEEP")
    PHYSACTIV  = code_select("Physical activity (NSCH code, higher = more days)", "PHYSACTIV")
    BEDTIME    = code_select("Consistent bedtime (NSCH code)", "BEDTIME")
    st.markdown("**Emotional well-being survey items (NSCH codes)**")
    K10Q11 = code_select("Survey item K10Q11", "K10Q11")
    K10Q12 = code_select("Survey item K10Q12", "K10Q12")
    K10Q13 = code_select("Survey item K10Q13", "K10Q13")
    K10Q14 = code_select("Survey item K10Q14", "K10Q14")
    BULLIED_R = code_select("Bullying victimization (NSCH code, 1 = lowest)", "BULLIED_R")

with right:
    st.subheader("Demographics & Health")
    SC_AGE_YEARS = st.slider("Age (years)", 12, 17, 15)
    SC_SEX       = code_select("Sex", "SC_SEX", {1: "Male", 2: "Female"})
    SC_RACE_R    = code_select("Race/Ethnicity", "SC_RACE_R",
                               {1: "White", 2: "Black", 3: "American Indian / Alaska Native",
                                4: "Asian", 5: "Native Hawaiian / Pacific Islander", 7: "Multiple races"})
    SC_HISPANIC_R= code_select("Hispanic origin", "SC_HISPANIC_R", {1: "Hispanic", 2: "Not Hispanic"})
    K2Q31A       = code_select("ADHD diagnosis", "K2Q31A", YES_NO)
    K2Q32A       = code_select("Anxiety diagnosis", "K2Q32A", YES_NO)
    FPL_I1       = st.slider("Household income as % of federal poverty level (FPL_I1)",
                             int(clean_df["FPL_I1"].min()), int(clean_df["FPL_I1"].max()), int(clean_df["FPL_I1"].median()))
    METRO_YN     = code_select("Metropolitan area", "METRO_YN", YES_NO)
    HHLANGUAGE   = code_select("Household language (NSCH code)", "HHLANGUAGE")

# =============================================================================
# SECTION 9: PREDICTION
# =============================================================================

st.markdown("---")
predict_btn = st.button("🔮 Assess Risk", use_container_width=True, type="primary")

if predict_btn:
    # Assemble raw inputs
    raw = {
        "SCREENTIME":    SCREENTIME,
        "HOURSLEEP":     HOURSLEEP,
        "PHYSACTIV":     PHYSACTIV,
        "BEDTIME":       BEDTIME,
        "K10Q11":        K10Q11,
        "K10Q12":        K10Q12,
        "K10Q13":        K10Q13,
        "K10Q14":        K10Q14,
        "SC_AGE_YEARS":  SC_AGE_YEARS,
        "SC_SEX":        SC_SEX,
        "SC_RACE_R":     SC_RACE_R,
        "SC_HISPANIC_R": SC_HISPANIC_R,
        "K2Q31A":        K2Q31A,
        "K2Q32A":        K2Q32A,
        "BULLIED_R":     BULLIED_R,
        "FPL_I1":        FPL_I1,
        "METRO_YN":      METRO_YN,
        "HHLANGUAGE":    HHLANGUAGE,
    }

    # Encode categoricals with the same LabelEncoders used in training
    enc = raw.copy()
    for col in CATEGORICAL_COLS:
        if col in encoders and col in enc:
            le = encoders[col]
            try:
                # Training encoded float codes ("1.0", "2.0"), so look up the same form.
                enc[col] = int(le.transform([str(float(raw[col]))])[0])
            except ValueError:
                # Should not happen now that the form only offers codes from the data.
                # Stop loudly instead of silently predicting on a wrong value.
                st.error(f"Value {raw[col]} for {col} was not seen in training data.")
                st.stop()

    # Build a DataFrame with features in the exact order the model expects
    input_df = pd.DataFrame([enc])[available_features]

    # Predict
    prediction   = clf.predict(input_df)[0]
    proba        = clf.predict_proba(input_df)[0]
    prob_dx      = proba[1]       # probability of depression diagnosis
    prob_no_dx   = proba[0]

    # Display
    st.subheader("Risk Assessment")
    res_col1, res_col2, res_col3 = st.columns(3)

    if prediction == 1:
        res_col1.error("🔴 **Elevated Risk**\nModel predicts: Depression Diagnosis")
    else:
        res_col1.success("🟢 **Lower Risk**\nModel predicts: No Diagnosis")

    res_col2.metric("Probability: Diagnosis",    f"{prob_dx:.1%}")
    res_col3.metric("Probability: No Diagnosis", f"{prob_no_dx:.1%}")

    # Risk bar
    st.progress(float(prob_dx), text=f"Depression risk score: {prob_dx:.1%}")

    # --- Top contributing features ---
    st.subheader("Key Factors for This Prediction")
    # Use feature importances as a proxy (global, not instance-level SHAP)
    importances = pd.Series(clf.feature_importances_, index=available_features)
    top5 = importances.sort_values(ascending=False).head(5)
    st.bar_chart(top5)
    st.caption(
        "Chart shows the five most influential features overall (model-wide importance, "
        "not specific to this individual prediction)."
    )

    with st.expander("Show full input sent to model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # Disclaimer
    st.info(
        "⚠️ This prediction is generated by a statistical model and is "
        "intended for research/educational use only. Always consult a licensed "
        "mental health professional for clinical assessment."
    )

# =============================================================================
# SECTION 10: GLOBAL FEATURE IMPORTANCE (always visible)
# =============================================================================

st.markdown("---")
st.subheader("Model: Global Feature Importances")
all_imp = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
st.bar_chart(all_imp)
st.caption(
    "These importances show which features the Random Forest used most "
    "across the training records. Anxiety diagnosis (K2Q32A) is by far the strongest "
    "predictor, followed by bullying (BULLIED_R) and ADHD diagnosis (K2Q31A)."
)

st.markdown("---")
st.caption("DSC 355 – Milestone 5 | Hali Sarah Parsons | November 2026 | Data: NSCH 2021")
