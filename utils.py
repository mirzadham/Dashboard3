# utils.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    try:
        df_2014 = pd.read_csv('2014.csv')
        df_2016 = pd.read_csv('2016.csv')
        return df_2014, df_2016
    except FileNotFoundError:
        st.error("ERROR: Make sure you have uploaded both '2014.csv' and '2016.csv'")
        st.stop()

@st.cache_data
def clean_and_merge_data(df_2014, df_2016):
    # Define mappings from original column names to standard names
    map_2014 = {
        'Age': 'age', 'Gender': 'gender', 'self_employed': 'self_employed',
        'family_history': 'family_history', 'treatment': 'treatment', 'work_interfere': 'work_interfere',
        'no_employees': 'no_employees', 'tech_company': 'tech_company', 'benefits': 'benefits',
        'care_options': 'care_options', 'wellness_program': 'wellness_program', 'seek_help': 'seek_help',
        'anonymity': 'anonymity', 'leave': 'leave', 'mental_health_consequence': 'mental_health_consequence',
        'coworkers': 'coworkers', 'supervisor': 'supervisor'
    }
    df_2014_clean = df_2014[list(map_2014.keys())].rename(columns=map_2014)

    map_2016 = {
        'What is your age?': 'age', 'What is your gender?': 'gender', 'Are you self-employed?': 'self_employed',
        'Do you have a family history of mental illness?': 'family_history',
        'Have you ever sought treatment for a mental health issue from a mental health professional?': 'treatment',
        'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?': 'work_interfere',
        'How many employees does your company or organization have?': 'no_employees',
        'Is your employer primarily a tech company/organization?': 'tech_company',
        'Does your employer provide mental health benefits as part of healthcare coverage?': 'benefits',
        'Do you know the options for mental health care available under your employer-provided coverage?': 'care_options',
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': 'wellness_program',
        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': 'seek_help',
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'anonymity',
        'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:': 'leave',
        'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'mental_health_consequence',
        'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'coworkers',
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'supervisor'
    }
    df_2016_clean = df_2016[list(map_2016.keys())].rename(columns=map_2016)

    df_merged = pd.concat([df_2014_clean, df_2016_clean], ignore_index=True)

    # Clean Age column
    df_merged['age'] = pd.to_numeric(df_merged['age'], errors='coerce')
    df_merged.dropna(subset=['age'], inplace=True)
    df_merged = df_merged[(df_merged['age'] >= 18) & (df_merged['age'] <= 75)].copy()
    df_merged['age'] = df_merged['age'].astype(int)

    # Clean Gender column
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "cis male"]
    female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
    trans_str = ["trans-female", "something akin to female...", "woman-ish", "transgender", "female (trans)"]

    def clean_gender(gender_str):
        if pd.isna(gender_str):
            return gender_str
        gender_str = str(gender_str).lower()
        if gender_str in male_str:
            return 'Male'
        elif gender_str in female_str:
            return 'Female'
        elif gender_str in trans_str:
            return 'Trans'
        else:
            return 'Other'

    df_merged['gender'] = df_merged['gender'].apply(clean_gender)

    # Impute missing values
    df_merged['self_employed'] = df_merged['self_employed'].fillna('No')
    df_merged['work_interfere'] = df_merged['work_interfere'].fillna('N/A')
    
    # Impute other categorical columns with mode
    categorical_cols = [
        'benefits', 'care_options', 'wellness_program', 'seek_help', 
        'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 
        'supervisor', 'no_employees', 'tech_company'
    ]
    for col in categorical_cols:
        df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0])

    # Standardize treatment column
    treatment_map = {'Yes': 1, 'No': 0, '1': 1, '0': 0, True: 1, False: 0}
    df_merged['treatment'] = df_merged['treatment'].astype(str).map(treatment_map)
    df_merged.dropna(subset=['treatment'], inplace=True)
    df_merged['treatment'] = df_merged['treatment'].astype(int)

    return df_merged

def setup_page():
    st.set_page_config(layout="wide", page_title="Mental Health in Tech Dashboard", page_icon="🧠")
    st.markdown("""
    <style>
        .reportview-container .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 1.5rem 1rem;
        }
        h1, h2, h3 {
            color: #1f77b4;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
        .stMetric {
            background-color: #0d3c75; /* Slightly darker navy */
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: white;
            border: 1px solid #1f77b4; /* Matches header color */
        }
        .stMetric h3 {
            font-weight: 700;
            font-size: 1.8rem;
            margin-bottom: 0.25rem;
        }
        .stMetric label {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        .stDataFrame {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
