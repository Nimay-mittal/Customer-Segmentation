import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

# ---------------------------
# Sidebar – Config
# ---------------------------
st.set_page_config(page_title="Credit Card Segmentation", layout="wide")
st.sidebar.title("Settings")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("CC GENERAL.csv")
    df.fillna(df.mean(), inplace=True)
    return df

df = load_data()

# ---------------------------
# Sidebar options
# ---------------------------
algo_choice = st.sidebar.selectbox("Select Clustering Algorithm", 
                                   ["KMeans", "Agglomerative", "Gaussian Mixture"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

# ---------------------------
# Data Preprocessing
# ---------------------------
X = df.drop("CUST_ID", axis=1, errors="ignore")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Clustering
# ---------------------------
if algo_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(X_scaled)

elif algo_choice == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df["Cluster"] = model.fit_predict(X_scaled)

else:  # Gaussian Mixture
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(X_scaled)

# ---------------------------
# PCA for visualization
# ---------------------------
pca = PCA(2)
pca_data = pca.fit_transform(X_scaled)
df["PCA1"] = pca_data[:,0]
df["PCA2"] = pca_data[:,1]

# ---------------------------
# Dashboard Layout
# ---------------------------
st.title("📊 Credit Card Customer Segmentation Dashboard")
st.markdown("This dashboard segments credit card customers based on spending and transaction behavior.")

# --- Top metrics
col1, col2 = st.columns(2)
col1.metric("Number of Customers", df.shape[0])
col2.metric("Number of Features", X.shape[1])

# --- Cluster Distribution
st.subheader("Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())

# --- PCA 2D Plot
st.subheader("Customer Segments (2D PCA Projection)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", ax=ax)
st.pyplot(fig)

# --- Cluster Profiles
st.subheader("Cluster Profiles (Mean Feature Values)")
cluster_summary = df.groupby("Cluster").mean().T
st.dataframe(cluster_summary)

# --- Heatmap
st.subheader("Cluster Feature Comparison")
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.heatmap(cluster_summary, cmap="coolwarm", annot=True, ax=ax2)
st.pyplot(fig2)

# --- Explore Individual Cluster
st.subheader("Explore a Cluster")
selected_cluster = st.selectbox("Choose a Cluster", df["Cluster"].unique())
st.write(df[df["Cluster"] == selected_cluster].head())
