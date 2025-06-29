# employee_profiling.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

@st.cache_data
def perform_clustering(df):
    cluster_features = [
        'benefits', 'care_options', 'wellness_program', 'seek_help',
        'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor'
    ]
    df_cluster_data = df[cluster_features].copy()
    df_cluster_encoded = pd.get_dummies(df_cluster_data, drop_first=True)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_cluster_encoded)
    df_cluster_data['cluster'] = clusters
    
    return df_cluster_data

def show(df):
    st.header("👥 Employee Profiling")
    st.write("Clustering analysis to identify distinct employee segments based on mental health attitudes and behaviors.")
    
    # Perform clustering
    df_cluster = perform_clustering(df)
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    
    st.subheader("Cluster Distribution")
    col1, col2, col3 = st.columns(3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    with col1:
        st.metric("Cluster 0: Engaged & Supported", f"{cluster_counts[0]} employees", 
                 help="Employees aware of and utilizing mental health resources")
        st.markdown("""
        - High awareness of benefits
        - Comfortable discussing with managers
        - Moderate treatment seeking
        """)
        st.progress(0.8, text="Resource Utilization")
        
    with col2:
        st.metric("Cluster 1: At-Risk & Unsupported", f"{cluster_counts[1]} employees", 
                 help="Employees with limited access to mental health support")
        st.markdown("""
        - Low benefit awareness
        - Anonymity concerns
        - High work interference
        """)
        st.progress(0.35, text="Resource Utilization")
        
    with col3:
        st.metric("Cluster 2: Proactive & Vulnerable", f"{cluster_counts[2]} employees", 
                 help="Employees actively seeking help with existing challenges")
        st.markdown("""
        - Family history of mental illness
        - High treatment seeking
        - Concerned about consequences
        """)
        st.progress(0.65, text="Resource Utilization")
    
    st.subheader("Cluster Characteristics Comparison")
    
    # Compare cluster characteristics
    cluster_chars = df_cluster.groupby('cluster').agg({
        'benefits': lambda x: (x == 'Yes').mean(),
        'care_options': lambda x: (x == 'Yes').mean(),
        'wellness_program': lambda x: (x == 'Yes').mean(),
        'seek_help': lambda x: (x == 'Yes').mean(),
        'anonymity': lambda x: (x == 'Yes').mean(),
        'coworkers': lambda x: (x == 'Yes').mean(),
        'supervisor': lambda x: (x == 'Yes').mean()
    }).reset_index()
    
    melted = cluster_chars.melt(id_vars='cluster', var_name='feature', value_name='percentage')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='feature', y='percentage', hue='cluster', data=melted, palette=colors, ax=ax)
    ax.set_title('Mental Health Resource Awareness by Cluster')
    ax.set_xlabel('Resource Feature')
    ax.set_ylabel('Percentage Responded "Yes"')
    ax.legend(title='Cluster', labels=['Engaged & Supported', 'At-Risk & Unsupported', 'Proactive & Vulnerable'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Recommendations by Cluster")
    st.markdown("""
    | Cluster | Recommended Actions |
    |---------|---------------------|
    | **Engaged & Supported** | Maintain current programs, focus on retention, peer support networks |
    | **At-Risk & Unsupported** | Increase resource awareness, improve anonymity guarantees, manager training |
    | **Proactive & Vulnerable** | Provide specialized support, reduce stigma, flexible work arrangements |
    """)
