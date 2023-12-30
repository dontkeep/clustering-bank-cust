import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Credit Card Customer Data.csv')

st.title('Segmentation Data Credit Card Customer Data Using K-Means Clustering')
st.write('This web app is made to do clustering on credit card customer data from a bank using K-Means Clustering method. This data is obtained from Kaggle.com which contains Avg_Credit_Limit, Total_Credit_Cards, Total_visits_online, Total_calls_made, and Customer Key.')
st.subheader('Raw Data')
st.write(df)

#add x
x = df.iloc[:,[2,3]]

st.subheader('Finding Optimum Number of Clusters with Elbow Method')

kmeans = KMeans()
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, "bx-")
plt.xlabel("Number of Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow = st.pyplot()




def k_means(n):
    kmeans = KMeans(n_clusters=n).fit(x)
    clusters_kmeans = kmeans.labels_
    df["cluster"] = clusters_kmeans

    plt.figure(figsize=(8, 6))

    for label in set(clusters_kmeans):
        cluster_data = x[clusters_kmeans == label]
        plt.scatter(cluster_data['Avg_Credit_Limit'], cluster_data['Total_Credit_Cards'], label=f"Cluster {label}", alpha=0.8)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Cluster Centers', marker='*')
    plt.xlabel("Avg_Credit_Limit")
    plt.ylabel("Total_Credit_Cards")
    plt.legend()
    plt.title('Clusters of Customers')
    st.subheader('Final Result of Dataset after Clustering')
    st.pyplot()
    st.write('From the clustering results, it can be seen that there are 3 clusters of customers. The first cluster is customers who have a low credit limit and a few credit cards, the second cluster is customers who have a medium credit limit and a medium number of credit cards, and the third cluster is customers who have a high credit limit and a lot of credit cards.')
    st.write(df)


st.subheader('Determine Number of Clusters')
cluster = st.slider('Please Determine Number of Clusters', 1, 10) 
k_means(cluster)
st.subheader('Summary')
st.write('The first cluster is customers who have a low credit limit and a few credit cards. This cluster has an average credit limit of 3370.10 and an average number of credit cards of 4.52. The second cluster is customers who have a medium credit limit and a medium number of credit cards. This cluster has an average credit limit of 14104.96 and an average number of credit cards of 5.49. The third cluster is customers who have a high credit limit and a lot of credit cards. This cluster has an average credit limit of 33782.38 and an average number of credit cards of 5.52.')