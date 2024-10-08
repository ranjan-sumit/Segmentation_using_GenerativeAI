# import streamlit as st
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# st.sidebar.title('Kinesso Auto-Campaign')

# # Function to calculate ELBO (for demonstration, using KMeans inertia as a proxy)
# def calculate_elbo(X, max_clusters):
#     elbow = []
#     for i in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=i)
#         kmeans.fit(X)
#         elbow.append(kmeans.inertia_)
#     return elbow

# # Main app
# st.title("Kinesso Auto-Campaign Toolkit")

# # Step 1: Upload CSV
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("Data Preview:")
#     st.dataframe(df.head())

#     # Step 2: Select columns for clustering
#     columns = st.multiselect("Select columns for clustering", options=df.columns.tolist())
    
#     if columns:
#         # Convert selected columns to a numeric format
#         X = df[columns].select_dtypes(include=[np.number])
        
#         # Step 3: Generate ELBO curve and Silhouette score
#         max_clusters = st.slider("Select maximum number of clusters for ELBO calculation", 1, 10)
#         if st.button("Generate ELBO Curve and Silhouette Score"):
#             elbo_values = calculate_elbo(X, max_clusters)
#             silhouette_values = []
#             for i in range(2, max_clusters + 1):
#                 kmeans = KMeans(n_clusters=i)
#                 kmeans.fit(X)
#                 silhouette_values.append(silhouette_score(X, kmeans.labels_))
            
#             # Plot ELBO
#             fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#             sns.lineplot(x=range(1, max_clusters + 1), y=elbo_values, ax=ax[0], marker='o')
#             ax[0].set_title('ELBO Curve')
#             ax[0].set_xlabel('Number of Clusters')
#             ax[0].set_ylabel('ELBO (Inertia)')
            
#             # Plot Silhouette Score
#             sns.lineplot(x=range(2, max_clusters + 1), y=silhouette_values, ax=ax[1], marker='o', color='orange')
#             ax[1].set_title('Silhouette Score')
#             ax[1].set_xlabel('Number of Clusters')
#             ax[1].set_ylabel('Silhouette Score')
            
#             st.pyplot(fig)
        
#         # Step 4: Set number of clusters and run algorithm
#         n_clusters = st.number_input("Enter number of clusters", min_value=1, max_value=10, value=3)
#         if st.button("Run Clustering Algorithm"):
#             kmeans = KMeans(n_clusters=n_clusters)
#             df['Cluster'] = kmeans.fit_predict(X)  # Cluster entire dataset
#             st.success("Clustering algorithm ran successfully!")
            
#             # Display the entire clustered dataset
#             st.write("Clustered Data:")
#             st.dataframe(df)

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_openai import OpenAI

# Sidebar for user inputs
st.sidebar.title('Kinesso Auto-Campaign')

# Step 1: Input OpenAI API Key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Function to calculate ELBO (for demonstration, using KMeans inertia as a proxy)
def calculate_elbo(X, max_clusters):
    elbow = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        elbow.append(kmeans.inertia_)
    return elbow

# Function to generate marketing strategies for each cluster
def generate_marketing_strategies(cluster_num, cluster_data, llm):
    # Prepare the input data for the prompt
    cluster_summary = f"This is Cluster {cluster_num} containing {len(cluster_data)} customers."
    
    # Create a descriptive summary for the cluster
    average_income = cluster_data['Income'].mean()
    predominant_relationship_status = cluster_data['In_relationship'].mode()[0]  # Assuming 'In_relationship' is a column
    
    template = """
    Given the following details about a customer cluster:
    - {cluster_summary}
    - Average Income: {average_income}
    - Predominant Relationship Status: {predominant_relationship_status}
    
    Please suggest analysis of cluster, what they buy more or how was previous campaign for them successful or not. What should be marketing strategy. Explain in brief in 100  words for each cluster and business language.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["cluster_summary", "average_income", "predominant_relationship_status"])
    filled_prompt = prompt.format(
        cluster_summary=cluster_summary,
        average_income=average_income,
        predominant_relationship_status=predominant_relationship_status
    )

    # Generate marketing strategies using the LLM
    llm_prediction = llm(filled_prompt)
    return llm_prediction

# Main app
st.title("Kinesso Auto-Campaign Toolkit")


tab1, tab2 = st.tabs(["Segmentation Insight", "Kinesso LLM Q & A"])

with tab1:
    st.header("Segmentation Insight")


    # Step 2: Upload CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
    
        # Step 3: Select columns for clustering
        columns = st.multiselect("Select columns for clustering", options=df.columns.tolist())
        
        if columns:
            # Convert selected columns to a numeric format
            X = df[columns].select_dtypes(include=[np.number])
            
            # Step 4: Generate ELBO curve and Silhouette score
            max_clusters = st.slider("Select maximum number of clusters for ELBO calculation", 1, 10)
            if st.button("Generate ELBO Curve and Silhouette Score"):
                elbo_values = calculate_elbo(X, max_clusters)
                silhouette_values = []
                for i in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i)
                    kmeans.fit(X)
                    silhouette_values.append(silhouette_score(X, kmeans.labels_))
                
                # Plot ELBO
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.lineplot(x=range(1, max_clusters + 1), y=elbo_values, ax=ax[0], marker='o')
                ax[0].set_title('ELBO Curve')
                ax[0].set_xlabel('Number of Clusters')
                ax[0].set_ylabel('ELBO (Inertia)')
                
                # Plot Silhouette Score
                sns.lineplot(x=range(2, max_clusters + 1), y=silhouette_values, ax=ax[1], marker='o', color='orange')
                ax[1].set_title('Silhouette Score')
                ax[1].set_xlabel('Number of Clusters')
                ax[1].set_ylabel('Silhouette Score')
                
                st.pyplot(fig)
            
            # Step 5: Set number of clusters and run algorithm
            n_clusters = st.number_input("Enter number of clusters", min_value=1, max_value=10, value=3)
            if st.button("Run Clustering Algorithm"):
                kmeans = KMeans(n_clusters=n_clusters)
                df['Cluster'] = kmeans.fit_predict(X)  # Cluster entire dataset
                st.success("Clustering algorithm ran successfully!")
                
                # Display the entire clustered dataset
                st.write("Clustered Data:")
                st.dataframe(df)
    
                # Check if API key is provided before generating insights
                if openai_api_key:
                    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
                    n_clusters = df['Cluster'].nunique()  # Get the number of unique clusters
                    
                    # Collect and display marketing strategies for each cluster
                    for i in range(n_clusters):
                        cluster_data = df[df['Cluster'] == i]
                        if cluster_data.empty:
                            st.write(f"No data available for Cluster {i}.")
                        else:
                            strategies = generate_marketing_strategies(i, cluster_data, llm)
                            st.write(f"\n**Marketing Strategies for Cluster {i}:**")
                            st.write(strategies)
                else:
                    st.warning("Please enter your OpenAI API Key to generate marketing strategies.")

with tab2:
    st.header("Kinesso LLM Q & A")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True,allow_dangerous_code=True)
        agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4o"),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,allow_dangerous_code=True
    )
        

    
