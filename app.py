# app.py
import streamlit as st
from utils import setup_page, load_data, clean_and_merge_data
from data_overview import show as show_data_overview
from descriptive_analytics import show as show_descriptive
from diagnostic_analytics import show as show_diagnostic
from predictive_analytics import show as show_predictive
from employee_profiling import show as show_profiling

# Initialize page
setup_page()

# Load data
df_2014, df_2016 = load_data()
df = clean_and_merge_data(df_2014, df_2016)

# Sidebar navigation
st.sidebar.title("Mental Health Dashboard")
sections = {
    "📊 Data Overview": show_data_overview,
    "📈 Employee Demographics Overview": show_descriptive,
    "🔍 Mental Wellness Correlations": show_diagnostic,
    "🤖 Mental Health Risk Assessment": show_predictive,
    "👥 Employee Profiling": show_profiling
}
selected = st.sidebar.radio("Navigate to:", list(sections.keys()))

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dataset Information**  
- **Records**: {len(df)} employees  
- **Years**: 2014 & 2016  
- **Variables**: 16  
""")

st.sidebar.markdown("---")
st.sidebar.caption("""
Created by [Group 4]  
[GitHub Repository](https://github.com/mirzadham/Despression-Dashboard/tree/main)  
""")

# Main content area
st.title("🧠 Mental Health in Tech Workplace Analysis")
sections[selected](df)  # Call the selected section function

# Footer
st.markdown("---")
st.caption("© 2023 Mental Health in Tech Dashboard | Data Sources: 2014 & 2016 OSMI Mental Health in Tech Surveys")
