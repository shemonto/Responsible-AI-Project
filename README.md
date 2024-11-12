# Predicting Cross-sell Insurance: Responsible AI Project
## Python, Scikit-learn, Pandas, Numpy, Imbalanced-learn, Streamlit

## Objective
Develop a machine learning model to predict customer interest in vehicle insurance based on their existing health insurance. This project emphasizes Responsible AI by ensuring transparency and interpretability through SHAP and LIME. An interactive dashboard is built using Streamlit to visualize model predictions, making model behavior more transparent for end-users.

---

## Dataset
The dataset simulates real-world data and includes features such as customer demographics, insurance details, and historical interactions. Key features include:

- **Gender, Age, Driving_License, Region_Code, Previously_Insured**
- **Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel, Vintage**
- **Response** (target variable indicating interest in vehicle insurance)

---

## Project Workflow

### Step 1: Data Preprocessing
1. **Data Loading**: Loaded data from `cross_sell_rai.csv`.
2. **Feature Encoding**: Encoded categorical variables for `Gender`, `Vehicle_Damage`, and `Vehicle_Age`.
3. **Balancing Classes**: Used `RandomOverSampler` from `imblearn` to address class imbalance in the target variable.
4. **Data Splitting and Scaling**:
   - Split the balanced dataset into training and testing sets with an 80-20 ratio.
   - Scaled the features using `StandardScaler` to standardize the data.

### Step 2: Model Development
1. **Model Choice**: The primary model used is `ExtraTreesClassifier`, selected for its interpretability and suitability for tabular data.
2. **Model Training**: The model was trained on the preprocessed data and saved to `st.session_state` to avoid reloading.

### Step 3: Model Interpretability
To ensure Responsible AI principles, SHAP and LIME were integrated into the project.

#### SHAP (SHapley Additive exPlanations):
- Initialized `shap.Explainer` with `feature_perturbation="interventional"` for more realistic feature impact estimation.
- Generated SHAP values for each instance to understand the contribution of individual features to the prediction.

#### LIME (Local Interpretable Model-agnostic Explanations):
- Initialized `lime.lime_tabular.LimeTabularExplainer`, which provides explanations for individual predictions by creating local surrogate models.
- Configured `LimeTabularExplainer` to use discretized features and interpret prediction probabilities across different features.

### Step 4: Interactive Dashboard
The project features an interactive Streamlit dashboard that allows users to view and interpret model predictions. Key functionalities include:

- **Model Prediction**:
  - Displays prediction results for a chosen instance, along with class probabilities.

- **SHAP Explanation**:
  - **Summary Plot**: Shows the distribution of feature impacts across the test dataset, allowing for a high-level view of feature importance.
  - **Waterfall Plot**: Provides a detailed breakdown of feature contributions for a specific instance.

- **LIME Explanation**:
  - **LIME Plot**: Visualizes feature importance for a single instance to highlight factors influencing the model's decision.
  - **Feature Importance Table**: Displays a ranked list of feature contributions based on LIME for enhanced interpretability.

### Step 5: Model Evaluation
The model was evaluated using the **F1-score**, achieving an impressive score of **0.987**. Performance was further validated with metrics such as **Precision** and **Recall** to ensure robust classification. **AUC** and **confusion matrix** plots were also generated to assess model discriminative power.

---

## Achievements
- **Ranked 2nd**: Achieved a top F1-score of 0.987 among all submissions in the Responsible AI project hosted by HiCounsellor.
- **Model Transparency**: Integrated SHAP and LIME to make the model explainable and interpretable for stakeholders, aligning with Responsible AI standards.
- **Streamlit Dashboard**: Delivered an interactive, transparent interface for model predictions and explanations.

---

## Outcome
The project has provided significant insights into customer behavior, enhancing cross-sell strategies and demonstrating a comprehensive understanding of applied machine learning and Responsible AI principles. The approach used in this project solidifies a foundation in explainable AI, feature engineering, and model evaluation for real-world applications.

---
