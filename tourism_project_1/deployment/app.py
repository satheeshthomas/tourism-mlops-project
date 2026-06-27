# Token comes from HF Space secret (set in Space settings — no hardcoded value)
import os, joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, login

# FIX: set_page_config is the FIRST Streamlit command
st.set_page_config(page_title="Tourism Package Predictor", page_icon="🌴", layout="wide")

MODEL_REPO = "SatheeshThomas/tourism-prj-model-1"
MODEL_FILE = "best-tourism-model-v2.joblib"

# Exact 18 features — same order as prep.py
FEATURES = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, token=token)
    return joblib.load(path)

model = load_model()

st.title("🌴 Tourism Package Purchase Predictor")
st.markdown("Fill in the customer details below to predict package purchase likelihood.")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    Age                      = st.number_input("Age", 18, 90, 35)
    CityTier                 = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch          = st.number_input("Duration of Pitch (mins)", 1, 60, 15)
    NumberOfPersonVisiting   = st.number_input("Persons Visiting", 1, 10, 2)
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 10, 0)
    NumberOfTrips            = st.number_input("Trips per Year", 0, 22, 2)
with col2:
    TypeofContact  = st.selectbox("Type of Contact", [0, 1],
                       format_func=lambda x: "Company Invited" if x==0 else "Self Enquiry")
    Occupation     = st.selectbox("Occupation", [0, 1, 2, 3],
                       format_func=lambda x: ["Free Lancer","Large Business","Self Employed","Salaried"][x])
    Gender         = st.selectbox("Gender", [0, 1],
                       format_func=lambda x: "Female" if x==0 else "Male")
    MaritalStatus  = st.selectbox("Marital Status", [0, 1, 2],
                       format_func=lambda x: ["Divorced","Married","Unmarried"][x])
    Designation    = st.selectbox("Designation", [0, 1, 2, 3, 4],
                       format_func=lambda x: ["AVP","Executive","Manager","Senior Manager","VP"][x])
    MonthlyIncome  = st.number_input("Monthly Income (₹)", 5000, 100000, 40000, step=1000)
with col3:
    ProductPitched         = st.selectbox("Product Pitched", [0, 1, 2, 3, 4],
                               format_func=lambda x: ["Basic","Deluxe","King","Multi","Super Deluxe"][x])
    PreferredPropertyStar  = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    NumberOfFollowups      = st.number_input("Number of Follow-ups", 1, 6, 3)
    Passport               = st.selectbox("Has Passport?", [0, 1],
                               format_func=lambda x: "No" if x==0 else "Yes")
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar                 = st.selectbox("Owns a Car?", [0, 1],
                               format_func=lambda x: "No" if x==0 else "Yes")

st.divider()
if st.button("🔍 Predict Purchase", use_container_width=True):
    input_df = pd.DataFrame([[
        Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
        Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
        PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
        PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting,
        Designation, MonthlyIncome
    ]], columns=FEATURES)

    proba = model.predict_proba(input_df)[0][1]
    pred  = model.predict(input_df)[0]

    st.markdown("### Prediction Result")
    if pred == 1:
        st.success(f"✅ Likely to Purchase — Confidence: {proba:.1%}")
    else:
        st.warning(f"❌ Unlikely to Purchase — Confidence: {1 - proba:.1%}")
    st.progress(float(proba), text=f"Purchase probability: {proba:.1%}")
