# diagnostic_analytics.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show(df):
    st.header("🔍 Mental Wellness Correlations")
    
    tab1, tab2 = st.tabs(["Treatment Analysis", "Work Interference"])
    
    with tab1:
        st.subheader("Treatment Seeking Behavior")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Gender**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='gender', hue='treatment', data=df, palette='Set2', ax=ax)
            ax.set_title('Treatment Seeking by Gender')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
            st.pyplot(fig)
            
        with col2:
            st.write("**By Company Size**")
            fig, ax = plt.subplots(figsize=(10, 6))
            company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
            sns.countplot(x='no_employees', hue='treatment', data=df, 
                         order=company_size_order, palette='viridis', ax=ax)
            ax.set_title('Treatment Seeking by Company Size')
            ax.set_xlabel('Number of Employees')
            ax.set_ylabel('Count')
            ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Treatment by Family History")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='family_history', hue='treatment', data=df, palette='coolwarm', ax=ax)
        ax.set_title('Treatment Seeking by Family History')
        ax.set_xlabel('Family History of Mental Illness')
        ax.set_ylabel('Count')
        ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Work Interference Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Work Interference Distribution**")
            filtered_df = df[df['work_interfere'] != 'N/A']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='work_interfere', data=filtered_df, 
                         order=['Never', 'Rarely', 'Sometimes', 'Often'], 
                         palette='magma', ax=ax)
            ax.set_title('Work Interference Levels')
            ax.set_xlabel('Interference Frequency')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        with col2:
            st.write("**Interference by Benefits Availability**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='benefits', hue='work_interfere', data=filtered_df, 
                         palette='viridis', ax=ax,
                         hue_order=['Never', 'Rarely', 'Sometimes', 'Often'])
            ax.set_title('Work Interference by Mental Health Benefits')
            ax.set_xlabel('Benefits Provided')
            ax.set_ylabel('Count')
            ax.legend(title='Interference Level')
            st.pyplot(fig)
        
        st.subheader("Interference by Company Size")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='no_employees', hue='work_interfere', data=filtered_df, 
                     order=company_size_order, palette='coolwarm', ax=ax,
                     hue_order=['Never', 'Rarely', 'Sometimes', 'Often'])
        ax.set_title('Work Interference by Company Size')
        ax.set_xlabel('Number of Employees')
        ax.set_ylabel('Count')
        ax.legend(title='Interference Level')
        plt.xticks(rotation=45)
        st.pyplot(fig)
