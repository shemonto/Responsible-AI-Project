import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import RandomOverSampler

print('prothom a asi')
# Set up Streamlit page config
st.set_page_config(page_title="Model Interpretability Dashboard", layout="wide")

# Load and preprocess data (run only once)
def load_and_preprocess_data():
    data = pd.read_csv('./cross_sell_rai.csv')
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    data['Vehicle_Age'] = data['Vehicle_Age'].map(vehicle_age_mapping)

    # Prepare feature matrix and target vector
    X = data[['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
    y = data['Response']

    # Balance the classes using RandomOverSampler
    over_sampling = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = over_sampling.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, data.columns

# Check if data has been preprocessed in session_state
if 'data_preprocessed' not in st.session_state:
    print('preprocess nai')
    st.session_state.X_train_scaled, st.session_state.X_test_scaled, st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.feature_names = load_and_preprocess_data()
    st.session_state.data_preprocessed = True 

# Check if model is loaded
if 'model' not in st.session_state:
    print('model nai')
    model = ExtraTreesClassifier(random_state=30)
    model.fit(st.session_state.X_train_scaled, st.session_state.y_train)
    st.session_state.model = model  # Store model in session_state

# Check if SHAP explainer is loaded
if 'shap_explainer' not in st.session_state:
    print('SHAP nai')
    explainer_shap = shap.Explainer(st.session_state.model, st.session_state.X_train_scaled, feature_perturbation="interventional")
    st.session_state.shap_explainer = explainer_shap

# Check if LIME explainer is loaded
if 'lime_explainer' not in st.session_state:
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        st.session_state.X_train_scaled,
        feature_names=st.session_state.feature_names,
        class_names=[0, 1],
        discretize_continuous=True
    )
    st.session_state.lime_explainer = lime_explainer

# Streamlit UI for user input and options
st.title("Interactive Model Interpretability Dashboard")
st.write("Choose a display option to view model predictions, SHAP, or LIME explanations for a specific instance.")

# Sidebar for display options
st.sidebar.title("Options")
display_option = st.sidebar.radio(
    "Display Option",
    ["SHAP Explanation", "LIME Explanation", "Model Prediction and Feature Values"]
)

# Sidebar for entering the instance index
instance_idx_input = st.sidebar.text_input("Enter Instance Index", value="0")

# Handle the instance index input
try:
    instance_idx = int(instance_idx_input)
    if instance_idx < 0:
        st.sidebar.warning("Index cannot be negative. Setting to 0.")
        instance_idx = 0
    elif instance_idx >= len(st.session_state.X_test_scaled):
        st.sidebar.warning(f"Index cannot exceed {len(st.session_state.X_test_scaled) - 1}. Setting to {len(st.session_state.X_test_scaled) - 1}.")
        instance_idx = len(st.session_state.X_test_scaled) - 1
except ValueError:
    st.sidebar.warning("Invalid input. Please enter a valid integer index.")
    instance_idx = 0

# Main Content Area based on selected option
if display_option == "Model Prediction and Feature Values":
    st.write(f"### Prediction for Instance {instance_idx}")
    prediction_proba = st.session_state.model.predict_proba([st.session_state.X_test_scaled[instance_idx]])[0]
    prediction = "Yes" if st.session_state.model.predict([st.session_state.X_test_scaled[instance_idx]])[0] == 1 else "No"
    
    # Display prediction and probability
    st.write(f"Prediction: **{prediction}**")
    st.write(f"Probability of interest: **{prediction_proba[1]:.2f}**")
    
    # Display feature values for the selected instance
    st.write("### Feature Values")
    selected_instance = st.session_state.X_test.iloc[instance_idx]
    for feature, value in selected_instance.items():
        st.write(f"{feature}: {value}")

elif display_option == "SHAP Explanation":
    st.subheader("SHAP Analysis")
    shap_values = st.session_state.shap_explainer.shap_values(st.session_state.X_test_scaled, approximate=True)

    # Display SHAP plots side by side
    col1, col2 = st.columns(2)
    
    # SHAP Summary Plot in the first column
    with col1:
        st.write("### SHAP Summary Plot")
        fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, st.session_state.X_test_scaled, feature_names=st.session_state.feature_names, show=False)
        st.pyplot(fig_summary)

    # SHAP Waterfall Plot in the second column
    with col2:
        st.write(f"### SHAP Waterfall Plot for Instance {instance_idx}")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[1][instance_idx],
                base_values=st.session_state.shap_explainer.expected_value[1],
                data=st.session_state.X_test_scaled[instance_idx],
                feature_names=st.session_state.feature_names
            )
        )
        st.pyplot(fig_waterfall)

elif display_option == "LIME Explanation":
    st.subheader("LIME Analysis")
    
    # Generate LIME explanation for the selected instance
    explanation_lime = st.session_state.lime_explainer.explain_instance(
        st.session_state.X_test_scaled[instance_idx], st.session_state.model.predict_proba, num_features=11
    )

    # Display the LIME explanation and feature importance side by side with spacing
    col1, col_space, col2 = st.columns([1, 0.2, 1])

    # LIME Explanation Plot in the first column
    with col1:
        st.write(f"### LIME Explanation Plot for Instance {instance_idx}")
        
        # Create the LIME plot
        fig_lime = explanation_lime.as_pyplot_figure()
        
        # Resize the plot to make it larger
        fig_lime.set_size_inches(14, 14)  # Adjust the size (width, height)
        
        # Access the axes and adjust font size
        for ax in fig_lime.get_axes():
            ax.tick_params(axis='both', labelsize=30)  # Increase font size for ticks (both x and y)
            ax.set_xlabel(ax.get_xlabel(), fontsize=36)  # Increase font size for x-axis label
            ax.set_ylabel(ax.get_ylabel(), fontsize=36)  # Increase font size for y-axis label
        
        # Set the title font size
        fig_lime.suptitle('Local Explanation for Class 1', fontsize=34)

        # Show the plot
        st.pyplot(fig_lime)

    # Show instance features and predictions on the second column
    with col2:
        st.write(f"### Instance {instance_idx} Features")
        st.write("Feature Values:")
        selected_instance = st.session_state.X_test.iloc[instance_idx]
        for feature, value in selected_instance.items():
            st.write(f"{feature}: {value}")
        
        st.write("Prediction:")
        st.write(f"Probability for Class 0: {prediction_proba[0]:.2f}")
        st.write(f"Probability for Class 1: {prediction_proba[1]:.2f}")
