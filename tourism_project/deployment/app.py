
import streamlit as st
import pandas as pd
import joblib


# Loading the trained model
MODEL_PATH = "tourism_project/deployment/best_model.pkl"
model = joblib.load(MODEL_PATH)


# Page title
st.title("Wellness Tourism Package Prediction")

st.write(
    "Enter the customer details below to predict "
    "whether the customer is likely to purchase the package."
)


# Customer inputs

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=20,
    value=3
)

Passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=10
)


# Prediction

if st.button("Predict"):

    # Store user inputs in a dataframe
    input_data = pd.DataFrame({
        "Age": [Age],
        "TypeofContact": [TypeofContact],
        "CityTier": [CityTier],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "NumberOfPersonVisiting": [NumberOfPersonVisiting],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "MaritalStatus": [MaritalStatus],
        "NumberOfTrips": [NumberOfTrips],
        "Passport": [Passport],
        "OwnCar": [OwnCar],
        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
        "Designation": [Designation],
        "MonthlyIncome": [MonthlyIncome],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "ProductPitched": [ProductPitched],
        "NumberOfFollowups": [NumberOfFollowups],
        "DurationOfPitch": [DurationOfPitch]
    })

    st.subheader("Customer Information")
    st.dataframe(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the "
            "Wellness Tourism Package."
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )
