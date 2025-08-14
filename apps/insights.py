import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

def insights_page():
    
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except FileNotFoundError:
        st.error("Employee dataset not found in `data/` folder.")
        return

    tab_visuals, tab_ai  = st.tabs(["Visual Insights", "AI Chat"])

    with tab_visuals:
        st.caption("Explore AI insights and visual patterns in employee attrition.")
        
        # Attrition by Department
        fig_dept = px.bar(
            df.groupby(['Department', 'Attrition']).size().reset_index(name='Count'),
            x='Department', 
            y='Count',
            color='Attrition',
            text='Count',
            barmode='group',
            title='Attrition by Department'
        )
        fig_dept.update_traces(texttemplate='%{text}', textposition='outside')
        fig_dept.update_layout(height=500)
        st.plotly_chart(fig_dept, use_container_width=True)

        # Age Distribution by Attrition Status
        fig_age = px.histogram(
            df,
            x='Age',
            color='Attrition',
            nbins=20,
            marginal='box',
            opacity=0.7,
            title='Age Distribution by Attrition Status'
        )
        fig_age.update_layout(height=500, barmode='overlay')
        st.plotly_chart(fig_age, use_container_width=True)

        # Employee Tenure Distribution
        fig_years = px.histogram(
            df,
            x='YearsAtCompany',
            color='Attrition',
            nbins=20,
            marginal='violin',
            opacity=0.7,
            title='Employee Tenure Distribution'
        )
        fig_years.update_layout(height=500, barmode='overlay')
        st.plotly_chart(fig_years, use_container_width=True)

        # Attrition Rate by Job Role
        role_attrition = df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)
        role_attrition['Total'] = role_attrition.sum(axis=1)
        role_attrition['Attrition_Rate'] = (role_attrition.get('Yes',0) / role_attrition['Total']) * 100
        role_attrition = role_attrition.sort_values('Attrition_Rate', ascending=False)

        fig_role = px.bar(
            role_attrition.reset_index(),
            x='Attrition_Rate',
            y='JobRole',
            orientation='h',
            text='Attrition_Rate',
            color='Attrition_Rate',
            title='Attrition Rate by Job Role'
        )
        fig_role.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_role.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_role, use_container_width=True)

        # High Risk Groups
        risk_factors = {
            'Overtime Workers': df[df['OverTime'] == 'Yes']['Attrition'].value_counts(normalize=True).get('Yes',0) * 100,
            'Low Job Satisfaction': df[df['JobSatisfaction'] == 1]['Attrition'].value_counts(normalize=True).get('Yes',0) * 100,
            'Long Commute (>20km)': df[df['DistanceFromHome'] > 20]['Attrition'].value_counts(normalize=True).get('Yes',0) * 100,
            'New Employees (<2 years)': df[df['YearsAtCompany'] < 2]['Attrition'].value_counts(normalize=True).get('Yes',0) * 100,
        }
        fig_risk = go.Figure(go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            text=[f'{v:.1f}%' for v in risk_factors.values()],
            textposition='outside'
        ))
        fig_risk.update_layout(title='High Risk Groups - Attrition Rates', xaxis_title='Attrition Rate (%)', height=400)
        st.plotly_chart(fig_risk, use_container_width=True)


    with tab_ai:
        API_KEY = st.secrets['MY_API_KEY']
        genai.configure(api_key=API_KEY)

        # initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # clear chat
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

        # chat input
        if user_input := st.chat_input("Ask a question about employee attrition:"):
            # save user's message
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                model = genai.GenerativeModel("gemma-3n-e4b-it")
                system_instruction = (
                    "You are an HR analytics assistant. "
                    "Answer questions related to human resources, employee attrition, workplace trends, "
                    "and HR best practices. Provide clear, actionable insights, reasoning, and data-driven suggestions when possible."
                )
                prompt = f"{system_instruction}\n\nDataset preview:\n{df.head().to_string()}\n\nUser question: {user_input}"

                with st.spinner("Generating AI response..."):
                    response = model.generate_content(prompt)
                    ai_text = response.text

            except Exception as e:
                ai_text = f"An error occurred: {e}"

            # save AI response
            st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

        # display chat messages
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(chat["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(chat["content"])


