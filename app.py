import streamlit as st
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="GPA Predictor", layout="centered")

st.title("🎓 Student GPA Prediction")
st.write("Enter student details to predict GPA")

# -------- USER INPUTS -------- #
age = st.number_input("Age", min_value=10, max_value=25, value=17)
study_time = st.number_input("Study Time per Week (hours)", min_value=0, max_value=60, value=10)
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=3)

gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox(
    "Ethnicity",
    ["African American", "Asian", "Caucasian", "Other"]
)

parent_edu = st.selectbox(
    "Parental Education",
    ["Bachelor's", "High School", "Higher", "None", "Some College"]
)

tutoring = st.selectbox("Tutoring", ["Yes", "No"])

parent_support = st.selectbox(
    "Parental Support",
    ["None", "Low", "Moderate", "High", "Very High"]
)

extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sports = st.selectbox("Sports", ["Yes", "No"])
music = st.selectbox("Music", ["Yes", "No"])
volunteering = st.selectbox("Volunteering", ["Yes", "No"])

all_activities = st.selectbox(
    "Overall Activities Level",
    ["No activity", "One activity", "Two activities", "Three activities", "All Activities"]
)

# -------- BUILD INPUT DICTIONARY -------- #
input_data = {col: 0 for col in feature_names}

# Numeric
input_data["Age"] = age
input_data["StudyTimeWeekly"] = study_time
input_data["Absences"] = absences

# Gender
input_data[f"Gender_{gender}"] = 1

# Ethnicity
input_data[f"Ethnicity_{ethnicity}"] = 1

# Parental Education
input_data[f"ParentalEducation_{parent_edu}"] = 1

# Tutoring
input_data[f"Tutoring_{tutoring}"] = 1

# Parental Support
input_data[f"ParentalSupport_{parent_support}"] = 1

# Activities
input_data[f"Extracurricular_{extracurricular}"] = 1
input_data[f"Sports_{sports}"] = 1
input_data[f"Music_{music}"] = 1
input_data[f"Volunteering_{volunteering}"] = 1
input_data[f"AllActivities_{all_activities}"] = 1


def get_grade_class(gpa):
    if gpa >= 3.5:
        return 0, "A"
    elif gpa >= 3.0:
        return 1, "B"
    elif gpa >= 2.5:
        return 2, "C"
    elif gpa >= 2.0:
        return 3, "D"
    else:
        return 4, "F"

# -------- PREDICTION -------- #
if st.button("🔮 Predict GPA"):
    x_new = pd.DataFrame([input_data], columns=feature_names)
    x_scaled = scaler.transform(x_new)
    gpa_pred = model.predict(x_scaled)[0]

    grade_id, grade_letter = get_grade_class(gpa_pred)

    st.success(f"📘 Predicted GPA: **{gpa_pred:.2f}**")
    st.info(f"🏆 Grade Class: **{grade_letter}** (Class {grade_id})")

