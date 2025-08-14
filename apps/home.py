import streamlit as st

def home_page():
    st.title("RetentionIQ")

    
    st.write(
        "RetentionIQ uses advanced machine learning to predict employee attrition (turnover, churn, reduction) "
        "risk before it happens. It empowers HR teams to take proactive steps to "
        "retain talent and improve workplace culture."
    )
    
    st.markdown("""
    <style>
    /* Target the next image after this style tag */
    img {
        max-height: 300px;
        width: 100%;
        object-fit: cover;
    }
    </style>
    """, unsafe_allow_html=True )

    st.image("images/employees.jpg", use_container_width=True)

    st.markdown(
    """
    <div style="text-align: center; font-size: 0.8em; color: gray; margin-bottom: 30px;">
    Photo by <a href="https://unsplash.com/@jopwell?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash" target="_blank">The Jopwell Collection</a> 
    on <a href="https://unsplash.com/photos/a-group-of-people-sitting-around-a-table-Y4uoL2SIGUQ?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash" target="_blank">Unsplash</a>
    </div>
    """,
    unsafe_allow_html=True )
    

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Tech Used")
        st.markdown("""
        - **Data Processing**: Pandas, NumPy, Scikit-learn, SMOTE 
        - **Visualization**: Plotly, Streamlit  
        - **Machine Learning**: XGBoost, LightGBM, CatBoost, Random Forest 
        """)

    with col2:
        st.markdown("#### Use Cases")
        st.markdown("""
        - **Proactive Retention**: Identify at-risk employees before they decide to leave  
        - **Strategic Planning**: Understand department-level attrition patterns  
        - **Budget Optimization**: Reduce recruitment and training costs  
        - **Culture Improvement**: Address systemic issues affecting retention  
        """)

    



      