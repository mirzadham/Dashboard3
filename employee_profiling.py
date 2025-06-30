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
    
    # Define cluster names and colors
    CLUSTER_NAMES = ["Engaged & Supported", "At-Risk & Unsupported", "Proactive & Vulnerable"]
    CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
    cluster_name_map = {i: name for i, name in enumerate(CLUSTER_NAMES)}
    color_map = {name: color for name, color in zip(CLUSTER_NAMES, CLUSTER_COLORS)}
    
    # Perform clustering
    df_cluster = perform_clustering(df)
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    
    # Map cluster numbers to names
    df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_name_map)
    
    st.subheader("Cluster Distribution")
    col1, col2, col3 = st.columns(3)
    
    for i, col in enumerate([col1, col2, col3]):
        cluster_name = CLUSTER_NAMES[i]
        count = cluster_counts[i]
        color = CLUSTER_COLORS[i]
        
        with col:
            st.metric(f"Cluster {i}: {cluster_name}", f"{count} employees", 
                     help="Employees aware of and utilizing mental health resources")
            
            # Cluster-specific descriptions
            if i == 0:
                st.markdown("""
                - High awareness of benefits
                - Comfortable discussing with managers
                - Moderate treatment seeking
                """)
                st.progress(0.8, text="Resource Utilization")
            elif i == 1:
                st.markdown("""
                - Low benefit awareness
                - Anonymity concerns
                - High work interference
                """)
                st.progress(0.35, text="Resource Utilization")
            else:
                st.markdown("""
                - Family history of mental illness
                - High treatment seeking
                - Concerned about consequences
                """)
                st.progress(0.65, text="Resource Utilization")
    
    st.subheader("Cluster Characteristics Comparison")
    
    # Compare cluster characteristics
    cluster_chars = df_cluster.groupby('cluster_name').agg({
        'benefits': lambda x: (x == 'Yes').mean(),
        'care_options': lambda x: (x == 'Yes').mean(),
        'wellness_program': lambda x: (x == 'Yes').mean(),
        'seek_help': lambda x: (x == 'Yes').mean(),
        'anonymity': lambda x: (x == 'Yes').mean(),
        'coworkers': lambda x: (x == 'Yes').mean(),
        'supervisor': lambda x: (x == 'Yes').mean()
    }).reset_index()
    
    melted = cluster_chars.melt(id_vars='cluster_name', var_name='feature', value_name='percentage')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create plot with consistent color mapping
    sns.barplot(
        x='feature', 
        y='percentage', 
        hue='cluster_name', 
        data=melted, 
        palette=color_map,
        hue_order=CLUSTER_NAMES,
        ax=ax
    )
    
    ax.set_title('Mental Health Resource Awareness by Cluster')
    ax.set_xlabel('Resource Feature')
    ax.set_ylabel('Percentage Responded "Yes"')
    ax.legend(title='Employee Segments')
    plt.xticks(rotation=45)
    
    # Apply the exact colors to legend handles
    handles, labels = ax.get_legend_handles_labels()
    for i, handle in enumerate(handles):
        handle.set_color(CLUSTER_COLORS[i])
    
    st.pyplot(fig)
    
    st.subheader("Recommendations by Cluster")
    st.markdown("""
    | Cluster | Recommended Actions |
    |---------|---------------------|
    | **Engaged & Supported** | Maintain current programs, focus on retention, peer support networks |
    | **At-Risk & Unsupported** | Increase resource awareness, improve anonymity guarantees, manager training |
    | **Proactive & Vulnerable** | Provide specialized support, reduce stigma, flexible work arrangements |
    """)
