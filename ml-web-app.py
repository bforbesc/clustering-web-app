import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,  davies_bouldin_score
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

pio.templates.default = "ggplot2"

np.random.seed(69)

st.write("""
# ✂️Clustering cross-sectional data

## Table of contents
1. [Data description](#data-description)
1. [Number of clusters choice](#number-of-clusters-choice)
1. [Cluster visualization](#cluster-visualization)

The purpose of this app is to compute and plot clusters for your **cross-sectional data**.

I assume you have a CSV file with your data **already cleaned** (i.e., without missing values, duplicates or string variables).

The algorithms used are **KMeans** and **tSNE**:
- **KMeans** clustering method is an unsupervised machine learning technique used to identify clusters of data objects in a dataset.  It is an example of partitional clustering which divides data objects into non-overlapping groups.  In other words, no object can be a member of more than one cluster, and every cluster must have at least one object.
- The **t-distributed stochastic neighbor embedding (t-SNE)** is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two-dimensional map. It models each high-dimensional object by a two-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
""")

st.sidebar.header('User input features')

# Collects user input features into dataframe
file_object = st.sidebar.file_uploader("Upload your cleaned CSV file", type=["csv"])
sep_option = st.sidebar.selectbox(
    'Which seperator would you like to use?',
     [",", ";", ".", ":", "\s+"])

@st.cache(allow_output_mutation=True)
def read_csv_cache():
    df = pd.read_csv(file_object, sep=sep_option)
    return df

try:
    df = read_csv_cache()
except:
    st.error('Please upload a valid CSV file!')
    st.stop()

if df.shape[1] == 1:
    st.error('Please select the appropriate separator for your CSV!')
    st.stop()

st.sidebar.header('Preferred number of clusters')
n_clusters = st.sidebar.slider('Number of clusters', 2, 15, 2)

# SUBSECTION: DATA
st.subheader('Data description')

# Displays the dataframe head
if st.checkbox('Show data'):
    st.write(df)

if st.checkbox('Show descriptive statistics'):
    st.write(df.describe())

# Data processing
@st.cache
def data_processing(data):
    numerical=data.select_dtypes(include=np.number).columns
    categorical=data.select_dtypes(exclude=np.number).columns

    ct = ColumnTransformer([
            ("ohe", OneHotEncoder(sparse=False ), categorical)
        ], remainder='passthrough')
    df_enc=ct.fit_transform(data)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_enc)
    return numerical, categorical, df_scaled

numerical, categorical, df_scaled = data_processing(df)

# Plot selected variables
option = st.selectbox(
 'Which variable would you like to plot the distribution of?', df.columns)

fig = px.histogram(df[option], title=option.replace("_", " ").title())
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="", showgrid=False)
fig.update_yaxes(title_text="Number of observations")
st.plotly_chart(fig, use_container_width=True)

# SUBSECTION: CHOOSING NUMBER OF CLUSTERS
st.subheader('Number of clusters choice')

# Try different number of clusters up to 15
r=16
@st.cache
def cluster_choice(df_scaled):
    sse = []
    silhouette_coefficients = []
    DB_score= []
    for k in range(2, r):
         kmeans = KMeans(n_clusters=k, init='k-means++' )
         kmeans.fit(df_scaled)
         sse.append(kmeans.inertia_)
         score = silhouette_score(df_scaled, kmeans.labels_)
         silhouette_coefficients.append(score)
         score = davies_bouldin_score(df_scaled, kmeans.labels_)
         DB_score.append(score)

    return sse, silhouette_coefficients, DB_score

sse, silhouette_coefficients, DB_score = cluster_choice(df_scaled)

figure1, axis = plt.subplots(3, 1, figsize=(12,12))

st.write("""
Three commonly used methods to evaluate the appropriate number of clusters are:
1. *The elbow method*
2. *The silhouette coefficient*
3. *Davies Bouldin index*

These are often used as complementary evaluation techniques rather than one being preferred over the other. 

1. **Elbow Method**
To perform the *elbow method*, we run several KMeans, incrementing the number of clusters with each iteration, and record the sum of the squared errors (SSE).
The SSE continues to decrease as we increase the number of clusters. As more centroids are added, the distance from each point to its closest centroid will decrease.
There’s a sweet spot **where the SSE curve starts to bend** known as the elbow point. The x-value of this point is thought to be a reasonable trade-off between error and number of clusters. 
""")

# Plot 1: SSE - choose elbow
fig = px.line(x = range(2, r), y = sse)
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="Number fo clusters")
fig.update_yaxes(title_text="SSE", showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.write("""
2. **Silhoutte coefficient** 
The *silhouette coefficient* is a measure of cluster cohesion and separation. It quantifies how well a data point fits into its assigned cluster based on two factors:
- How close the data point is to other points in the cluster
- How far away the data point is from points in other clusters
Its values ranges from -1 and 1. **Larger numbers** indicate that samples are closer to their clusters than they are to other clusters.
In the ```sklearn``` implementation, the average *silhouette coefficient* of all the samples is summarized into one score.
""")

# Plot 2: Silhouette Coefficient - choose local maximum
fig = px.line(x = range(2, r), y = silhouette_coefficients)
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="Number fo clusters")
fig.update_yaxes(title_text="Silhouette coefficient", showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.write("""
3. **Davies-Bouldin index** 
The  *Davies-Bouldin index* is easier to calculation than the *silhouette scores*. 
It is the ratio between the within cluster distances and the between cluster distances, averaged across clusters. Its is therefore bounded between 0 and 1.
A **lower index** relates to a model with better separation between the clusters.
""")

# Plot 3: Davies Bouldin Score - choose local minimum
fig = px.line(x = range(2, r), y = DB_score)
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="Number fo clusters")
fig.update_yaxes(title_text="Davies Bouldin score", showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.write("""
> Please choose your desired number of clusters (based on your combined interpretation of the previous three methods/ graphs) on the slider bar on the left panel **Preferred number of clusters** ⬅️.
""")

# SUBSECTION: VISUALIZATION
st.subheader('Cluster visualization')

# KMeans
kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(df_scaled)

# Aggregate clusters
@st.cache
def categorical_groupby(lst):
    res_dct = {lst[i]: "count" for i in range(0, len(lst), 1)}
    return res_dct

@st.cache(allow_output_mutation=True)
def numerical_groupby(lst):
    res_dct = {lst[i]: "mean" for i in range(0, len(lst), 1)}
    return res_dct

categorical_agg = categorical_groupby(categorical)
numerical_agg = numerical_groupby(numerical)
numerical_agg.update(categorical_agg)

st.write("""
Below I plot your **Preferred number of clusters** using the **t-SNE** algorithm. 
""")

# t-SNE
tsne = TSNE(random_state=69,  perplexity = 30)
X_tsne = tsne.fit_transform(df_scaled)

# Plot 4: t-SNE
fig = px.scatter(x = X_tsne[:, 0], y = X_tsne[:, 1], color=kmeans.labels_, color_continuous_scale=px.colors.sequential.Agsunset_r)
fig.update_layout(showlegend=False)
fig.update_xaxes(title_text="t-SNE feature 1")
fig.update_yaxes(title_text="t-SNE feature 2")
fig.update(layout_coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

df["cluster"] = pd.Series(kmeans.labels_, index=df.index)
df_clustered = df.groupby("cluster").agg(numerical_agg)

if st.checkbox('Show clusters description'):
    st.write(df_clustered)
    st.write("""
    *Notes: numerical and categorical features aggregated using mean and number of observations, respectively; index corresponds to the cluster number.*
    """)

if st.checkbox('Show original data with cluster assignment'):
    st.write(df)
