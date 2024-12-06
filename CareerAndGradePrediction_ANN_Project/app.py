import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer  # Add this import



# Load both models
with open("model.pkl", "rb") as f:
    cgpa_model = pickle.load(f)

with open("career_model.pkl", "rb") as f:
    career_model = pickle.load(f)
    
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("ann_cgpa_model.pkl", "rb") as f:
    ann_cgpa_model = pickle.load(f)

with open("career_ann_model.pkl", "rb") as f:
    ann_career_model = pickle.load(f)
# Load the column means for handling missing values
with open("column_means.pkl", "rb") as file:
    column_means = pickle.load(file)
# Function for CGPA prediction
def predict_cgpa(input_data):
    df = pd.DataFrame(input_data, index=[0])
    # Handle missing values with training mean
    df.fillna(column_means, inplace=True)

    # Scale the input data
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    scaled_data = scaler.transform(df)

    prediction = cgpa_model.predict(df)
    prediction_ann = ann_cgpa_model.predict(scaled_data)
    return prediction[0],prediction_ann[0]

def predict_career(input_data):
    # Convert input to DataFrame directly, as the model handles encoding
    df = pd.DataFrame(input_data, index=[0])

    # NEW: Make the prediction
    prediction = career_model.predict(df)

    # Decode the predicted class back to the original label
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    prediction_ann = ann_career_model.predict(df)  # ANN model prediction
    predicted_class_ann = label_encoder.inverse_transform(prediction_ann)[0]
    return predicted_class,predicted_class_ann  # Return the decoded prediction


# Add custom top bar below Streamlit's native header
st.markdown(
    """
    <style>
    
    .custom-topbar {
        background-color: #4CAF50; /* Green background */
        padding: 15px; /* Padding for the top bar */
        font-size: 30px; /* Font size for the text */
        color: white; /* Text color */
        text-align: center; /* Center alignment for the text */
        font-weight: bold; /* Bold font */
        
    }
    </style>
    <div class="custom-topbar">
        Student Predictor Application
    </div>
    """,
    unsafe_allow_html=True
)
# Streamlit UI
#st.title("Student Predictor")

# Menu for selecting prediction type
option = st.selectbox("Choose Prediction Type:", ["CGPA Prediction", "Career Prediction"])

# CGPA Prediction form
if option == "CGPA Prediction":
    st.subheader("CGPA Prediction")
    input_data = {
        "Semester 1 GPA": st.number_input("Semester 1 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 2 GPA": st.number_input("Semester 2 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 3 GPA": st.number_input("Semester 3 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 4 GPA": st.number_input("Semester 4 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 5 GPA": st.number_input("Semester 5 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 6 GPA": st.number_input("Semester 6 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 7 GPA": st.number_input("Semester 7 GPA", min_value=0.0, max_value=4.0, step=0.10),
        "Semester 8 GPA": st.number_input("Semester 8 GPA", min_value=0.0, max_value=4.0, step=0.10),
    }

    if st.button("Predict CGPA"):
    # Replace any empty input with None for missing values handling
        input_data_cleaned = {k: (v if v != 0.0 else None) for k, v in input_data.items()}
        prediction,prediction_ann = predict_cgpa(input_data_cleaned)
        st.success(f"Predicted CGPA (ML): {prediction:.2f}")
        st.success(f"Predicted CGPA (ANN): {prediction_ann:.2f}")
# Career Prediction form
elif option == "Career Prediction":
    st.subheader("Career Prediction")
    input_data = {
        "CGPA": st.number_input("CGPA", min_value=0.0, max_value=4.0, step=0.10),
        "Extra Curricular Activities": st.selectbox("Extra Curricular Activities", ["Football", "Cricket", "Coding Club", "Debating", "Music", "Art"]),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Shift": st.selectbox("Shift", ["Morning", "Evening"]),
        "University": st.selectbox("University", ["NUST", "COMSATS","University of Karachi","University of Punjab","IBA"])
    }

    if st.button("Predict Career Strength Area"):
        # Preprocess input data
        input_data_cleaned = {
            "CGPA": input_data["CGPA"],
            "Extra Curricular Activities": input_data["Extra Curricular Activities"],
            "Gender": input_data["Gender"],
            "Shift": input_data["Shift"],
             "University": input_data["University"],
        }
        
        # Use the updated predict_career function
        prediction,prediction_ann = predict_career(input_data_cleaned)
        st.success(f"Predicted Career Strength Area (ML): {prediction}")
        st.success(f"Predicted Career Strength Area (ANN): {prediction_ann}")
# Add footer with team members
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f3;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Developed By : Areeba Ejaz, Bismah Kulsoom, Farah Fatima, Kainat Iqbal</p>
    </div>
    """,
    unsafe_allow_html=True
)
