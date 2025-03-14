import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page title with custom styling
st.set_page_config(page_title="Readmission Prediction System", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')

model = load_model()

# Page title with enhanced styling
st.title("Patient Readmission Risk Prediction System")
st.write("Please input patient information for prediction")

# Create input form with two-column layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=65, help="Patient's age in years")
        frailty = st.number_input("Fried's Frailty Phenotype", min_value=0.0, max_value=1.0, value=0.08, format="%.3f", help="Frailty score between 0 and 1")
        vertebral = st.selectbox("Vertebral Fracture", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Presence of vertebral fracture")
        hospital_stay = st.number_input("Hospital Stay (days)", min_value=0, max_value=30, value=5, help="Length of hospital stay")
        falls = st.selectbox("History of Falls", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="History of falls")

    with col2:
        steadi = st.number_input("STEADI Score", min_value=0, max_value=10, value=3, help="STEADI fall risk score")
        weight_loss = st.selectbox("Weight Loss", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Recent weight loss")
        albumin = st.number_input("Albumin Level", min_value=20.0, max_value=60.0, value=40.0, format="%.1f", help="Albumin level in g/L")
        renal = st.selectbox("Renal Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Presence of renal disease")
        pulmonary = st.selectbox("Pulmonary Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Presence of pulmonary disease")

    # Submit button with custom styling
    submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Prepare input data
        input_data = np.array([[
            age, frailty, vertebral, hospital_stay, falls,
            steadi, weight_loss, albumin, renal, pulmonary
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Calculate SHAP values for the single input
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # Display results with enhanced styling
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error("Prediction: High Risk of Readmission")
            st.write(f"**Readmission Probability: {probability[0][1]:.2%}**")
        else:
            st.success("Prediction: Low Risk of Readmission")
            st.write(f"**Readmission Probability: {probability[0][1]:.2%}**")

        # Display SHAP Waterfall Plot with enhanced aesthetics
        st.markdown("---")
        st.subheader("Feature Contribution Analysis")
        
        # Create SHAP Waterfall Plot
        plt.figure(figsize=(12, 6))  # Adjusted size for better readability
        
        # Try to use seaborn style, fall back to default if it fails
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
        
        feature_names = ["Age", "Frailty Score", "Vertebral Fracture", "Hospital Stay", 
                         "Falls History", "STEADI Score", "Weight Loss", "Albumin Level",
                         "Renal Disease", "Pulmonary Disease"]
        
        # Get SHAP values for the positive class (class 1 for binary classification)
        shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        
        # Create a SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=input_data[0],
            feature_names=feature_names
        )
        
        # Plot Waterfall Plot with customization
        shap.waterfall_plot(explanation, show=False)
        
        # Customize plot aesthetics
        plt.title("Feature Contributions to Readmission Risk", fontsize=14, pad=10)
        plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)  # Add subtle grid for readability
        plt.tight_layout()
        
        # Add base value annotation
        plt.axvline(x=base_value, color='gray', linestyle='--', label=f'Base Value = {base_value:.2f}')
        plt.legend(fontsize=10, loc='upper right')
        
        # Display plot in Streamlit
        st.pyplot(plt)
        plt.close()

        # Display input summary with styled table
        st.markdown("---")
        st.subheader("Patient Information Summary")
        summary_data = {
            "Feature": feature_names,
            "Value": [age, frailty, "Yes" if vertebral == 1 else "No", 
                      hospital_stay, "Yes" if falls == 1 else "No",
                      steadi, "Yes" if weight_loss == 1 else "No", 
                      albumin, "Yes" if renal == 1 else "No",
                      "Yes" if pulmonary == 1 else "No"]
        }
        st.table(pd.DataFrame(summary_data).style.set_properties(**{'text-align': 'left'}).set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f0f0f0'), ('font-weight', 'bold')]}]
        ))

# Add sidebar information with enhanced styling
st.sidebar.title("About")
st.sidebar.info(
    "This is a machine learning-based readmission risk prediction system.\n\n"
    "The system uses XGBoost algorithm to help medical staff assess patient readmission risk.\n\n"
    "**Prediction factors include:**\n"
    "- Patient Age\n"
    "- Frailty Score\n"
    "- Vertebral Fracture\n"
    "- Hospital Stay\n"
    "- Falls History\n"
    "- STEADI Score\n"
    "- Weight Loss\n"
    "- Albumin Level\n"
    "- Renal Disease\n"
    "- Pulmonary Disease",
    icon="ℹ️"
)

# Add usage instructions with enhanced styling
st.sidebar.title("Instructions")
st.sidebar.info(
    "1. Fill in all required patient information\n"
    "2. Click 'Predict' button\n"
    "3. System will display readmission risk and probability\n"
    "4. Results are for reference only, please follow medical advice",
    icon="📋"
)
