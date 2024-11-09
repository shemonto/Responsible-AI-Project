import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer
import pickle
import streamlit as st
import shap
import lime
import lime.lime_tabular


data = pd.read_csv('./cross_sell_rai.csv')

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0}) 
data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'Yes':1, 'No':0}) 
vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
data['Vehicle_Age'] = data['Vehicle_Age'].map(vehicle_age_mapping)


x = data[['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
y = data['Response']


over_sampling = RandomOverSampler(random_state=0)
x_resampled, y_resampled = over_sampling.fit_resample(x, y)


X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, random_state=42, test_size=0.2)
feature_names = X_test.columns

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


et_model = ExtraTreesClassifier(random_state=30)
et_model.fit(X_train_scaled, y_train)

print('**** DAS ******')
st.set_page_config(page_title="Model Interpretability Dashboard", layout="wide")

# Title and Description
st.title("Interactive Model Interpretability Dashboard")
st.write("Choose an interpretability method (SHAP or LIME) and select a specific instance to analyze the model's prediction.")

# Sidebar Controls
st.sidebar.title("Options")
method = st.sidebar.radio("Choose Explanation Method", ["SHAP", "LIME"])

# Use a text input for the instance index, and convert the input to an integer
instance_idx_input = st.sidebar.text_input(
    "Enter Instance Index", value="0"
)

# Initialize instance_idx with a default value
instance_idx = 0

# Add an 'Enter' button to confirm input
enter_button = st.sidebar.button("Enter")

# Check if the button was clicked
if enter_button:
    try:
        # Convert the input to an integer
        instance_idx = int(instance_idx_input)
        # Ensure the index is within the valid range
        if instance_idx < 0:
            st.sidebar.warning("Index cannot be negative. Setting to 0.")
            instance_idx = 0
        elif instance_idx >= len(X_test_scaled):
            st.sidebar.warning(f"Index cannot exceed {len(X_test_scaled) - 1}. Setting to {len(X_test_scaled) - 1}.")
            instance_idx = len(X_test_scaled) - 1
    except ValueError:
        st.sidebar.warning("Invalid input. Please enter a valid integer index.")
        instance_idx = 0


# Main layout with columns
col1, col2 = st.columns(2)

# SHAP Explanation
if method == "SHAP":
    st.subheader("SHAP Analysis")

    # SHAP Explainer and SHAP values calculation
    explainer_shap = shap.Explainer(et_model, X_train_scaled, feature_perturbation="interventional")
    shap_values = explainer_shap.shap_values(X_test_scaled, approximate=True)

    # Display SHAP Summary Plot
    with col1:
        st.write("### SHAP Summary Plot")
        fig_summary, ax_summary = plt.subplots(figsize = (10, 6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        st.pyplot(fig_summary)

    # Display SHAP Waterfall Plot for Selected Instance
    with col2:
        st.write(f"### SHAP Waterfall Plot for Instance {instance_idx}")
        fig_waterfall, ax_waterfall = plt.subplots(figsize = (10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[1][instance_idx],
                base_values=explainer_shap.expected_value[1],
                data=X_test_scaled[instance_idx],
                feature_names=feature_names
            )
        )
        st.pyplot(fig_waterfall)

# LIME Explanation
elif method == "LIME":
    st.subheader("LIME Analysis")

    # LIME Explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=[0, 1],
        discretize_continuous=True
    )

    # Generate LIME Explanation for the selected instance
    explanation_lime = explainer_lime.explain_instance(
        X_test_scaled[instance_idx], et_model.predict_proba, num_features=11
    )

    # Display LIME Explanation Plot
    with col1:
        st.write(f"### LIME Explanation Plot for Instance {instance_idx}")
        fig_lime = explanation_lime.as_pyplot_figure()
        st.pyplot(fig_lime)

    # Textual Feature Importances from LIME
    with col2:
        st.write("### LIME Feature Importances")
        explanation_list = explanation_lime.as_list()
        for feature, importance in explanation_list:
            st.write(f"{feature}: {importance:.4f}")

# Footer notes
st.write("---")
st.write("""
This interactive dashboard allows you to toggle between SHAP and LIME explanations.
- **SHAP**: Provides global feature importance with a summary plot and a local explanation with a waterfall plot.
- **LIME**: Shows local feature importance explanations as a bar plot for the selected instance.
""")