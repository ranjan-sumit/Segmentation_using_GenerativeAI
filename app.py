# import streamlit as st
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # Function to calculate ELBO (for demonstration, using KMeans inertia as a proxy)
# def calculate_elbo(X, max_clusters):
#     elbow = []
#     for i in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=i)
#         kmeans.fit(X)
#         elbow.append(kmeans.inertia_)
#     return elbow

# # Main app
# st.title("Clustering App")

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

#             # Step 5: Merge clusters with master data
#             st.write("Merged Data with Cluster Assignments:")
#             st.dataframe(df)

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables for OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Function to calculate ELBO (for demonstration, using KMeans inertia as a proxy)
def calculate_elbo(X, max_clusters):
    elbow = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        elbow.append(kmeans.inertia_)
    return elbow

# Function to predict campaign success
def predict_campaign_success(customer_data, llm):
    # Feature Engineering
    customer_data['MntTotal'] = (
        customer_data['MntWines'] + 
        customer_data['MntFruits'] + 
        customer_data['MntMeatProducts'] + 
        customer_data['MntFishProducts'] + 
        customer_data['MntSweetProducts'] + 
        customer_data['MntGoldProds']
    )

    # Prepare data for predictions
    income_desc = "high" if customer_data['Income'] > 100000 else "moderate"
    recency_desc = "recently engaged" if customer_data['Recency'] < 30 else "less engaged"
    total_spent_desc = f"spent a total of ${customer_data['MntTotal']} on various products."
    
    template = """
    Given the following customer campaign response data:
    - Income: {income_desc}
    - Recency: {recency_desc}
    - Total Spent: {total_spent_desc}
    - Accepted previous campaigns: {accepted_campaigns}
    
    Please suggest in 100 words about personalized offers and engagement strategies to increase the likelihood of this customer accepting future campaigns. Focus on their preferences for wine and fruits, and consider their engagement level based on recency.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["income_desc", "recency_desc", "total_spent_desc", "accepted_campaigns"])
    filled_prompt = prompt.format(
        income_desc=income_desc,
        recency_desc=recency_desc,
        total_spent_desc=total_spent_desc,
        accepted_campaigns=customer_data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].tolist()
    )

    # Generate insights using the LLM
    llm_prediction = llm(filled_prompt)
    return llm_prediction

# Main app
st.title("Clustering App with Marketing Insights")

# Sidebar for OpenAI Key
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Step 2: Select columns for clustering
    columns = st.multiselect("Select columns for clustering", options=df.columns.tolist())
    
    if columns:
        # Convert selected columns to a numeric format
        X = df[columns].select_dtypes(include=[np.number])
        
        # Step 3: Generate ELBO curve and Silhouette score
        max_clusters = st.slider("Select maximum number of clusters for ELBO calculation", 1, 20)
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
        
        # Step 4: Set number of clusters and run algorithm
        n_clusters = st.number_input("Enter number of clusters", min_value=1, max_value=20, value=3)
        if st.button("Run Clustering Algorithm"):
            kmeans = KMeans(n_clusters=n_clusters)
            df['Cluster'] = kmeans.fit_predict(X)  # Cluster entire dataset
            st.success("Clustering algorithm ran successfully!")
            
            # Display the entire clustered dataset
            st.write("Clustered Data:")
            st.dataframe(df)

            # Step 5: Generate insights for each cluster
            if openai_api_key:
                llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
                
                if st.button("Generate Cluster Insights"):
                    for i in range(n_clusters):
                        cluster_data = df[df['Cluster'] == i]
                        if cluster_data.empty:
                            st.warning(f"No data available for Cluster {i}.")
                        else:
                            insights = predict_campaign_success(cluster_data.iloc[0], llm)  # Pass the first customer in the cluster
                            st.subheader(f"Generated Insights for Cluster {i}")
                            st.text(insights)

            # Step 6: Merge clusters with master data
            st.write("Merged Data with Cluster Assignments:")
            st.dataframe(df)
