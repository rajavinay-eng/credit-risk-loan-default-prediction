import os, json, re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Risk Demo", page_icon="🏦", layout="wide")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    b = joblib.load("models/credit_model.joblib")
    with open("models/feature_defaults.json") as f:
        defaults = json.load(f)
    return b, defaults

b, DEFAULTS = load_model()
model, FEATURES, CAT_COLS, CATS = b["model"], b["features"], b["cat_cols"], b["categories"]
GEMINI_MODEL = "gemini-flash-latest"

# Sensitive attributes are never shown as reasons
HIDDEN = {"DAYS_BIRTH", "age_years", "NAME_FAMILY_STATUS", "CODE_GENDER"}

NAMES = {
    "EXT_SOURCE_1": "External credit score 1 (0 to 1, higher is safer)",
    "EXT_SOURCE_2": "External credit score 2 (0 to 1, higher is safer)",
    "EXT_SOURCE_3": "External credit score 3 (0 to 1, higher is safer)",
    "inst_late_rate": "Share of past instalments paid late",
    "inst_underpaid_rate": "Share of past instalments underpaid",
    "inst_max_days_late": "Most days late on one instalment",
    "prev_approved_count": "Past Home Credit loans approved",
    "prev_refused_count": "Past Home Credit applications refused",
    "prev_refused_rate": "Share of past applications refused",
    "bureau_active_count": "Active loans with other lenders",
    "bureau_total_debt": "Total debt with other lenders",
    "bureau_oldest_loan_days": "Years since first loan with other lenders",
    "AMT_CREDIT": "Loan size compared with typical applicants",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "AMT_ANNUITY": "Annual payment",
    "AMT_INCOME_TOTAL": "Annual income",
    "DAYS_EMPLOYED": "Years in current job",
    "credit_to_income": "Loan amount divided by income",
    "annuity_to_income": "Annual payment divided by income",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education",
    "ORGANIZATION_TYPE": "Employer type",
    "OCCUPATION_TYPE": "Occupation",
    "OWN_CAR_AGE": "Age of own car (years)",
    "is_pensioner_or_unemployed": "Pensioner or unemployed (1 = yes)",
}

def fmt(f, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if f in ("DAYS_EMPLOYED", "bureau_oldest_loan_days"):
        return round(-float(v) / 365.25, 1)
    if f.endswith("_rate"):
        return f"{round(float(v) * 100, 1)}%"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return int(v) if v.is_integer() else round(v, 3)
    return str(v)

# ---------- Sidebar: made-up applicant ----------
st.sidebar.header("Made-up applicant")
st.sidebar.caption("Every other detail is set to a typical value.")

income = st.sidebar.number_input("Annual income", 25000, 900000, 150000, step=5000)
loan = st.sidebar.number_input("Loan amount", 45000, 4000000, 500000, step=10000)
payment = st.sidebar.number_input("Annual payment", 1600, 260000, 27000, step=1000)
ext1 = st.sidebar.slider("External credit score 1", 0.0, 1.0, 0.50, 0.01)
ext2 = st.sidebar.slider("External credit score 2", 0.0, 1.0, 0.57, 0.01)
ext3 = st.sidebar.slider("External credit score 3", 0.0, 1.0, 0.53, 0.01)
late = st.sidebar.slider("% of past instalments paid late", 0, 100, 2)
refused = st.sidebar.slider("% of past applications refused", 0, 100, 0)
active = st.sidebar.number_input("Active loans with other lenders", 0, 30, 2)

inc_types = CATS["NAME_INCOME_TYPE"]
income_type = st.sidebar.selectbox("Income type", inc_types,
                                   index=inc_types.index("Working") if "Working" in inc_types else 0)
edu_types = CATS["NAME_EDUCATION_TYPE"]
education = st.sidebar.selectbox("Education", edu_types)
no_job = income_type in ("Pensioner", "Unemployed")
years_job = st.sidebar.number_input("Years in current job", 0.0, 50.0, 4.0, 0.5, disabled=no_job)
age = st.sidebar.slider("Age (used by the model, never shown as a reason)", 21, 69, 40)

# ---------- Build one model row ----------
row = dict(DEFAULTS)
row.update({
    "AMT_INCOME_TOTAL": float(income), "AMT_CREDIT": float(loan),
    "AMT_ANNUITY": float(payment), "AMT_GOODS_PRICE": float(loan),
    "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
    "inst_late_rate": late / 100, "prev_refused_rate": refused / 100,
    "bureau_active_count": float(active),
    "NAME_INCOME_TYPE": income_type, "NAME_EDUCATION_TYPE": education,
    "DAYS_BIRTH": -age * 365.25, "age_years": float(age),
    "credit_to_income": loan / income, "annuity_to_income": payment / income,
    "is_pensioner_or_unemployed": int(no_job),
    "DAYS_EMPLOYED": np.nan if no_job else -years_job * 365.25,
})
if no_job:
    row["ORGANIZATION_TYPE"] = None

X = pd.DataFrame([row])[FEATURES]
for c in CAT_COLS:
    X[c] = pd.Categorical(X[c], categories=CATS[c])
num_cols = [c for c in FEATURES if c not in CAT_COLS]
X[num_cols] = X[num_cols].astype(float)

# ---------- Predict + explain ----------
p = float(model.predict_proba(X)[:, 1][0])
band = "DECLINE" if p >= 0.25 else "REVIEW" if p >= 0.12 else "APPROVE"

contrib = pd.Series(model.predict(X, pred_contrib=True)[0, :-1], index=FEATURES)
contrib = contrib.drop(labels=[f for f in HIDDEN if f in contrib.index])
up = contrib[contrib > 0].sort_values(ascending=False).head(3)
down = contrib[contrib < 0].sort_values().head(3)

# ---------- Page ----------
st.title("🏦 Credit Risk Demo")
st.caption("LightGBM trained on Home Credit data (Kaggle). Made-up applicants only. "
           "Not financial advice.")

c1, c2 = st.columns(2)
c1.metric("Predicted default risk", f"{p:.1%}")
with c2:
    st.write("**Recommendation**")
    if band == "APPROVE":
        st.success("APPROVE (risk under 12%)")
    elif band == "REVIEW":
        st.warning("HUMAN REVIEW (risk 12% to 25%)")
    else:
        st.error("DECLINE (risk 25% or more)")

r1, r2 = st.columns(2)
with r1:
    st.subheader("⬆ Raising risk")
    for f in up.index:
        st.write(f"- {NAMES.get(f, f)}: **{fmt(f, X.iloc[0][f])}**")
with r2:
    st.subheader("⬇ Lowering risk")
    for f in down.index:
        st.write(f"- {NAMES.get(f, f)}: **{fmt(f, X.iloc[0][f])}**")

# ---------- AI credit memo ----------
facts = {
    "default_risk_percent": round(p * 100, 1),
    "recommendation": band,
    "loan_amount": int(loan), "annual_income": int(income), "annual_payment": int(payment),
    "risk_increasing_factors": {NAMES.get(f, f): fmt(f, X.iloc[0][f]) for f in up.index},
    "risk_reducing_factors": {NAMES.get(f, f): fmt(f, X.iloc[0][f]) for f in down.index},
}

PROMPT = """You are a credit analyst assistant. Write a short credit memo (80 to 120 words) for a loan officer.
Rules:
- Use ONLY the facts in the JSON below. Do not add any number, fact, or assumption that is not in it.
- State the recommendation exactly as given: APPROVE, REVIEW, or DECLINE.
- Mention the default risk percent.
- Explain the main risk-increasing and risk-reducing factors in plain English.
- Do not mention gender, age, or marital status.
Facts:
{facts}"""

NUM = r"\d[\d,]*\.?\d*"

def allowed_numbers(obj):
    nums = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k); walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            nums.add(abs(float(x)))
        elif isinstance(x, str):
            for m in re.findall(NUM, x):
                nums.add(float(m.replace(",", "").rstrip(".")))
    walk(obj)
    return nums

def invented_numbers(memo, facts):
    allowed = allowed_numbers(facts)
    found = [float(m.replace(",", "").rstrip(".")) for m in re.findall(NUM, memo)]
    return [n for n in found if not any(abs(n - a) <= max(0.051, 0.001 * a) for a in allowed)]

def get_client():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)

st.divider()
st.subheader("🤖 AI credit memo")
with st.expander("Facts sent to the AI (it may use nothing else)"):
    st.json(facts)

if st.button("Write memo"):
    client = get_client()
    if client is None:
        st.info("No Gemini API key found, so the memo is switched off.")
    else:
        try:
            with st.spinner("Writing..."):
                memo = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=PROMPT.format(facts=json.dumps(facts, indent=2))).text
            st.markdown(memo)
            bad = invented_numbers(memo, facts)
            if bad:
                st.error(f"Fact check: numbers not in the fact sheet: {bad}")
            else:
                st.success("Fact check: every number in the memo comes from the fact sheet.")
        except Exception as e:
            st.warning(f"The AI service didn't respond (free-tier limits are common). Details: {str(e)[:150]}")

st.divider()
st.caption("Test results on 61,503 held-out applicants: ROC-AUC 0.771 (baseline 0.753). "
           "AI memo check: 0 invented numbers in 19 test memos.")