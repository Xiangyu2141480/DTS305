# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar for iterations
import warnings
import sys

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, precision_recall_fscore_support,
                             confusion_matrix)
from scipy.optimize import linear_sum_assignment

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Set random seed for reproducibility
np.random.seed(42)

# Read the data set
data1 = pd.read_csv("Data Set 1.csv", encoding="iso-8859-1")
data2 = pd.read_csv("Data Set 2.csv", encoding="utf-8")
data3 = pd.read_csv("Data Set 3.csv", encoding="utf-8")

# Number of printed data points
print("Number of data 1 points:", len(data1))
print("Number of data 2 points:", len(data2))
print("Number of data 3 points:", len(data3))

data1.head()

data2.head()

data3.head()

# Data Cleaning

# Information on data1 features
data1.info()

# Information on data2 features
data2.info()

# Information on data3 features
data3.info()

# To remove the NA values
data1 = data1.dropna()
print("The total number of data1-points after removing the rows with missing values are:", len(data1))

# To remove the NA values
data2 = data2.dropna()
print("The total number of data2-points after removing the rows with missing values are:", len(data2))

# To remove the NA values
data3 = data3.dropna()
print("The total number of data3-points after removing the rows with missing values are:", len(data3))

# explore the unique values in the categorical features to get a clear idea of the data1.
print("Total categories in the feature NewsType:\n", data1["NewsType"].value_counts(), "\n")

# explore the unique values in the categorical features to get a clear idea of the data2.
print("Total categories in the feature label:\n", data2["label"].value_counts(), "\n")

# explore the unique values in the categorical features to get a clear idea of the data3.
print("Total categories in the feature Price Sentiment:\n", data3["Price Sentiment"].value_counts(), "\n")

# ### Feature Processing

# View dataset column names
print(data1.columns)
print(data2.columns)
print(data3.columns)

# Create TF-IDF vector
vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)

# The correct column names are "Article", "Review Text" and "title"
dataset1_tfidf = vectorizer.fit_transform(data1["Article"])
dataset2_tfidf = vectorizer.fit_transform(data2["text"])
dataset3_tfidf = vectorizer.fit_transform(data3["News"])

# Convert to dense matrix
dataset1_dense = dataset1_tfidf.toarray()
dataset2_dense = dataset2_tfidf.toarray()
dataset3_dense = dataset3_tfidf.toarray()

datasets_dense = [dataset1_dense, dataset2_dense, dataset3_dense]
dataset_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']

# Dimensionality Reduction with PCA or TruncatedSVD
pca = PCA(n_components=3)
svd = TruncatedSVD(n_components=3)  # the 3 could be changed

# Apply PCA or SVD to each dataset
datasets_pca = []
for dataset_dense in datasets_dense:
    datasets_pca.append(pca.fit_transform(dataset_dense))  # Or use svd.fit_transform(dataset_dense)

# Range of k values to test
range_n_clusters = list(range(2, 11))
n_init = 10

for dataset_pca, dataset_name in zip(datasets_pca, dataset_names):
    silhouette_avg_scores = []

    print(f"Evaluating {dataset_name}...")

    # Progress bar for clustering iterations
    for n_clusters in tqdm(range_n_clusters, desc=f'Processing {dataset_name}'):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, random_state=42)
        kmeans.fit(dataset_pca)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(dataset_pca, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)

    # Plot silhouette score vs number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette score')
    plt.title(f'Silhouette Score vs Number of Clusters for {dataset_name}')
    plt.show()

    # Select the best k value
    best_k = range_n_clusters[np.argmax(silhouette_avg_scores)]

    # Run K-means with the best k value
    kmeans_best = KMeans(n_clusters=best_k, init='k-means++', n_init=n_init, random_state=42)
    kmeans_best.fit(dataset_pca)
    best_labels = kmeans_best.labels_

    # Calculate the silhouette score for the best k value
    best_silhouette_avg = silhouette_score(dataset_pca, best_labels)

    # Print the best k value and silhouette score
    print(f"{dataset_name}: Best number of clusters: {best_k}")
    print(f"{dataset_name}: Best silhouette score for k = {best_k}: {best_silhouette_avg}")

# ### Dataset 1: Best number of clusters: 4
# ### Dataset 2: Best number of clusters: 3
# ### Dataset 3: Best number of clusters: 4

# ### Results Printing

# True labels for each dataset
true_labels_dataset1 = data1['NewsType'].values
true_labels_dataset2 = data2['label'].values
true_labels_dataset3 = data3['Price Sentiment'].values

datasets_pca = []
for dataset_dense in datasets_dense:
    datasets_pca.append(pca.fit_transform(dataset_dense))

# Use LabelEncoder to convert string labels to numeric labels
label_encoders = [LabelEncoder(), LabelEncoder(), LabelEncoder()]
true_labels_encoded = [
    le.fit_transform(labels) for labels, le in
    zip([true_labels_dataset1, true_labels_dataset2, true_labels_dataset3], label_encoders)
]

datasets_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']
best_k_values = [4, 3, 4]  # These are the optimal K values for each dataset


# Function to align predicted clusters with true labels using Hungarian Algorithm
def align_labels(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # Maximize the alignment (negative for maximization)

    # Create a new predicted labels array with aligned labels
    aligned_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        aligned_labels[predicted_labels == j] = i

    return aligned_labels


# Run K-Means and compute metrics for each dataset
for datasets_pca, dataset_name, best_k, true_labels in zip(datasets_pca, datasets_names, best_k_values,
                                                           true_labels_encoded):
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(datasets_pca)
    predicted_labels = kmeans.labels_

    # Align predicted labels to true labels
    aligned_labels = align_labels(true_labels, predicted_labels)

    # Compute Precision, Recall, and F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, aligned_labels, average='weighted')

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, aligned_labels)

    # Print results
    print(f"{dataset_name}: Best number of clusters: {best_k}")
    print(f"{dataset_name}: Precision: {precision:.4f}")
    print(f"{dataset_name}: Recall: {recall:.4f}")
    print(f"{dataset_name}: F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in range(best_k)],
                yticklabels=[f'True {i}' for i in range(len(set(true_labels)))] if len(
                    set(true_labels)) < 20 else 'True labels')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

datasets_dense = [dataset1_dense, dataset2_dense, dataset3_dense]
datasets_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']

# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
datasets_pca = [pca.fit_transform(dataset_dense) for dataset_dense in datasets_dense]


# Function to create a 3D scatter plot
def plot_3d_clusters(dataset_pca, labels, dataset_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(dataset_pca[:, 0], dataset_pca[:, 1], dataset_pca[:, 2],
                         c=labels, cmap='viridis', alpha=0.5)

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(f'3D Clustering Visualization for {dataset_name}')

    plt.show()


# Run the plotting function for each dataset
best_k_values = [4, 3, 4]  # Optimal K values for each dataset
for dataset_pca, dataset_name, best_k in zip(datasets_pca, datasets_names, best_k_values):
    kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(dataset_pca)
    predicted_labels = kmeans.labels_
    plot_3d_clusters(dataset_pca, predicted_labels, dataset_name)
