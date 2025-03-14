import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Readmission Prediction System", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')

model = load_model()

# Load or create a background dataset for SHAP Summary Plot
# For demonstration, you can create a dummy dataset with similar features.
# In practice, you should use your training dataset or a representative subset.
@st.cache_data
def load_background_data():
    # Replace this with your actual training dataset or a subset
    np.random.seed(42)
    num_samples = 100  # Number of background samples for SHAP
    background_data = pd.DataFrame({
        "Age": np.random.randint(50, 90, num_samples),
        "Frailty Score": np.random.uniform(0, 1, num_samples),
        "Vertebral Fracture": np.random.choice([0, 1], num_samples),
        "Hospital Stay": np.random.randint(1, 30, num_samples),
        "Falls History": np.random.choice([0, 1], num_samples),
        "STEADI Score": np.random.randint(0, 10, num_samples),
        "Weight Loss": np.random.choice([0, 1], num_samples),
        "Albumin Level": np.random.uniform(20, 60, num_samples),
        "Renal Disease": np.random.choice([0, 1], num_samples),
        "Pulmonary Disease": np.random.choice([0, 1], num_samples)
    })
    return background_data

background_data = load_background_data()

# Page title
st.title("Patient Readmission Risk Prediction System")
st.write("Please input patient information for prediction")

# Create input form
with st.form("prediction_form"):
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        frailty = st.number_input("Fried's Frailty Phenotype", min_value=0.0, max_value=1.0, value=0.08, format="%.3f")
        vertebral = st.selectbox("Vertebral Fracture", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hospital_stay = st.number_input("Hospital Stay (days)", min_value=0, max_value=30, value=5)
        falls = st.selectbox("History of Falls", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        steadi = st.number_input("STEADI Score", min_value=0, max_value=10, value=3)
        weight_loss = st.selectbox("Weight Loss", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        albumin = st.number_input("Albumin Level", min_value=20.0, max_value=60.0, value=40.0, format="%.1f")
        renal = st.selectbox("Renal Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pulmonary = st.selectbox("Pulmonary Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data
        input_data = np.array([[
            age, frailty, vertebral, hospital_stay, falls,
            steadi, weight_loss, albumin, renal, pulmonary
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model, background_data)  # Use background data for SHAP
        shap_values = explainer.shap_values(background_data)  # Compute SHAP values for background data
        
        # Display results
        st.write("---")
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error("Prediction: High Risk of Readmission")
            st.write(f"Readmission Probability: {probability[0][1]:.2%}")
        else:
            st.success("Prediction: Low Risk of Readmission")
            st.write(f"Readmission Probability: {probability[0][1]:.2%}")

        # Display SHAP Summary Plot
        st.write("---")
        st.subheader("Feature Contribution Analysis (Global)")
        
        # Create SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        
        feature_names = ["Age", "Frailty Score", "Vertebral Fracture", "Hospital Stay", 
                         "Falls History", "STEADI Score", "Weight Loss", "Albumin Level",
                         "Renal Disease", "Pulmonary Disease"]
        
        # Use shap.summary_plot for the background dataset
        shap.summary_plot(
            shap_values[1] if isinstance(shap_values, list) else shap_values,  # Use class 1 SHAP values for binary classification
            background_data,
            feature_names=feature_names,
            show=False
        )
        
        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

        # Display input summary
        st.write("---")
        st.subheader("Patient Information Summary")
        summary_data = {
            "Feature": feature_names,
            "Value": [age, frailty, "Yes" if vertebral == 1 else "No", 
                      hospital_stay, "Yes" if falls == 1 else "No",
                      steadi, "Yes" if weight_loss == 1 else "No", 
                      albumin, "Yes" if renal == 1 else "No",
                      "Yes" if pulmonary == 1 else "No"]
        }
        st.table(pd.DataFrame(summary_data))

# Add sidebar information
st.sidebar.title("About")
st.sidebar.info(
    "This is a machine learning-based readmission risk prediction system.\n\n"
    "The system uses XGBoost algorithm to help medical staff assess patient readmission risk.\n\n"
    "Prediction factors include:\n"
    "- Patient Age\n"
    "- Frailty Score\n"
    "- Vertebral Fracture\n"
    "- Hospital Stay\n"
    "- Falls History\n"
    "- STEADI Score\n"
    "- Weight Loss\n"
    "- Albumin Level\n"
    "- Renal Disease\n"
    "- Pulmonary Disease"
)

# Add usage instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Fill in all required patient information\n"
    "2. Click 'Predict' button\n"
    "3. System will display readmission risk and probability\n"
    "4. Results are for reference only, please follow medical advice"
)
