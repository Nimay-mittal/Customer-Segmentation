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
    df.loc[(df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].median()
    df.loc[(df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT'] = df['CREDIT_LIMIT'].median()
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
data_imputed= pd.DataFrame(X_scaled, columns= X.columns)
best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
data_final = pd.DataFrame(data_imputed[best_cols])

# ---------------------------
# Clustering
# ---------------------------
if algo_choice == "KMeans":
    model = KMeans(n_clusters=n_clusters)
    df["Cluster"] = model.fit_predict(data_final)

elif algo_choice == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df["Cluster"] = model.fit_predict(data_final)

else:  # Gaussian Mixture
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(data_final)

# ---------------------------
# PCA for visualization
# ---------------------------
pca = PCA(2)
pca_data = pca.fit_transform(data_imputed)
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
cluster_summary = df.groupby("Cluster")[df.select_dtypes(include="number").columns].mean().T
st.dataframe(cluster_summary)

# --- Heatmap
st.subheader("Cluster Feature Comparison")

# Ensure only numeric cluster summary is used
numeric_summary = df.groupby("Cluster")[df.select_dtypes(include="number").columns].mean().T
normalized = (numeric_summary - numeric_summary.min()) / (numeric_summary.max() - numeric_summary.min())

fig2, ax2 = plt.subplots(figsize=(12,6))
sns.heatmap(normalized, cmap="coolwarm", annot=False, ax=ax2)
st.pyplot(fig2)


# --- Explore Individual Cluster
st.subheader("Explore a Cluster")
selected_cluster = st.selectbox("Choose a Cluster", df["Cluster"].unique())
st.write(df[df["Cluster"] == selected_cluster].head())










