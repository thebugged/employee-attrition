import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

API_KEY = st.secrets['MY_API_KEY']

genai.configure(api_key=API_KEY)

@st.cache_resource
def load_models():
    try:
        model_metadata = joblib.load('models/model_metadata.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        if model_metadata.get('model_type') == 'CatBoost':
            model = CatBoostClassifier()
            model.load_model('models/best_model_catboost.cbm')
            scaler = None
        else:
            model = joblib.load(f'models/best_model_{model_metadata["model_type"].lower()}.pkl')
            scaler = joblib.load('models/scaler.pkl')

        return model, scaler, label_encoders, model_metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def generate_ai_text(probability, employee_data):
    """Generate Risk Assessment & Recommended Actions using Gemma model."""
    try:
        model = genai.GenerativeModel('gemma-3n-e4b-it')
        prompt = f"""
        You are an HR attrition analysis assistant.
        Given the following employee data and attrition probability, provide:
        1. A concise Risk Assessment
        2. Specific Recommended Actions to address the situation.

        Attrition Probability: {probability:.2%}

        Employee Data:
        {employee_data.to_dict(orient='records')[0]}

        Output in this format:
        **Risk Assessment** ...
        **Recommended Actions:**
        - ...
        - ...
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI text: {e}"


def predict_page():
    st.caption("Enter employee details to predict attrition probability and get AI recommendations")
    st.markdown("")
    
    model, scaler, label_encoders, model_metadata = load_models()
    
    if model is None:
        st.error("Models not loaded. Please check the models directory.")
        return

    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    
    with col2:
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])
    
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col4:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", 18, 65, 35)
        education = st.selectbox("Education", 
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}[x])
    
    with col2:
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        percent_salary_hike = st.number_input("Last Salary Hike (%)", 0, 25, 12)
    
    with col3:
        years_at_company = st.number_input("Years at Company", 0, 40, 5)
        total_working_years = st.number_input("Total Working Years", 0, 40, 10)
    
    with col4:
        overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
    
    with col2:
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    
    with col3:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
    
    with col4:
        job_level = st.slider("Job Level", 1, 5, 2)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    
    # additional fields
    st.markdown("")
    with st.expander("Additional Information (Optional)"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            distance_from_home = st.number_input("Distance from Home (km)", 1, 30, 10)
            num_companies_worked = st.number_input("Previous Companies", 0, 10, 2)
        
        with col2:
            education_field = st.selectbox("Education Field",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
            training_times_last_year = st.number_input("Training Times Last Year", 0, 6, 2)
        
        with col3:
            years_in_current_role = st.number_input("Years in Current Role", 0, 20, 3)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 2)
        
        with col4:
            years_with_curr_manager = st.number_input("Years with Current Manager", 0, 20, 3)
            
            st.write("")
    
    st.markdown("")
    predict_button = st.button("Predict Attrition Risk", type="primary")
        
    
    # results section
    if predict_button:
        # prepare the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EducationField': [education_field],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'Gender': [gender],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MaritalStatus': [marital_status],
            'MonthlyIncome': [monthly_income],
            'NumCompaniesWorked': [num_companies_worked],
            'OverTime': [overtime],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager],
            'DailyRate': [800],
            'HourlyRate': [65],
            'MonthlyRate': [15000]
        })
        
        # encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col])
                except:
                    # handle unknown categories
                    input_data[col] = 0
        
        # make prediction
        try:
            if model_metadata.get('needs_scaling') and scaler is not None:
                input_scaled = scaler.transform(input_data)
                prediction_proba = model.predict_proba(input_scaled)[0]
            else:
                prediction_proba = model.predict_proba(input_data)[0]
            
            risk_probability = prediction_proba[1]
            
            # display results
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            with result_col1:
                st.markdown("#### Results")
            


            with result_col2:
                # risk gauge
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_probability * 100,
                    title={'text': "Attrition Risk Probability", 'font': {'size': 20}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "#90EE90"},
                            {'range': [33, 66], 'color': "#FFD700"},
                            {'range': [66, 100], 'color': "#FF6B6B"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_probability * 100
                        }
                    }
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # risk level and recommendations
            with st.spinner("Generating recommendations... Please wait"):
                ai_output = generate_ai_text(risk_probability, input_data)
            st.markdown(ai_output)
            
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all fields are filled correctly.")
    

if __name__ == "__main__":
    predict_page()