import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
file_path1 = 'Data Set 1.csv'
file_path2 = 'Data Set 2.csv'
file_path3 = 'Data Set 3.csv'

# Attempt to read the datasets with error replacement for encoding issues
dataset1 = pd.read_csv(file_path1, encoding='ISO-8859-1')
dataset2 = pd.read_csv(file_path2, encoding='ISO-8859-1')
dataset3 = pd.read_csv(file_path3, encoding='ISO-8859-1')

# Display the first few rows of the datasets to check the structure
print("Number of data 1 points:", dataset1.shape)
print("Number of dataset2 points:", dataset2.shape)
print("Number of dataset3 points:", dataset3.shape)

# Information and description of datasets
dataset1.info()
dataset1.describe()
dataset2.info()
dataset2.describe()
dataset3.info()

# Create TF-IDF vector for each dataset
vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
dataset1_tfidf = vectorizer.fit_transform(dataset1["Article"])
dataset2_tfidf = vectorizer.fit_transform(dataset2["text"])
dataset3_tfidf = vectorizer.fit_transform(dataset3["News"])

# Convert to dense matrix
dataset1_dense = dataset1_tfidf.toarray()
dataset2_dense = dataset2_tfidf.toarray()
dataset3_dense = dataset3_tfidf.toarray()

# Step 1: Dimensionality reduction using TruncatedSVD
n_components = 3  # Adjust based on the dataset's dimensionality
svd = TruncatedSVD(n_components=n_components)
dataset1_reduced = svd.fit_transform(dataset1_dense)
dataset2_reduced = svd.fit_transform(dataset2_dense)
dataset3_reduced = svd.fit_transform(dataset3_dense)

print("Dataset 1 reduced shape:", dataset1_reduced.shape)
print("Dataset 2 reduced shape:", dataset2_reduced.shape)
print("Dataset 3 reduced shape:", dataset3_reduced.shape)

# Create a dictionary of datasets
datasets = {
    'Dataset 1': dataset1['Article'],
    'Dataset 2': dataset2['text'],
    'Dataset 3': dataset3['News']
}

# Real labels dictionary
true_labels_dict = {
    'Dataset 1': dataset1['NewsType'].values,
    'Dataset 2': dataset2['label'].values,
    'Dataset 3': dataset3['Price Sentiment'].values
}

# Vectorizing text using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
vectorized_data = {dataset_name: vectorizer.fit_transform(texts) for dataset_name, texts in datasets.items()}

# Perplexity evaluation for different numbers of topics
topic_range = range(2, 11)
perplexity_scores = {}

for dataset_name, X_text in vectorized_data.items():
    perplexities = []
    for n_topics in topic_range:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X_text)
        perplexity = lda.perplexity(X_text)
        perplexities.append(perplexity)
    perplexity_scores[dataset_name] = perplexities

# Visualize perplexity for each dataset
for dataset_name, perplexities in perplexity_scores.items():
    plt.figure(figsize=(8, 6))
    plt.plot(topic_range, perplexities, marker='o')
    plt.title(f'Perplexity vs Number of Topics for {dataset_name}')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.xticks(topic_range)
    plt.grid(True)
    plt.show()

# Label encoding the true labels
label_encoder = LabelEncoder()
for dataset_name in true_labels_dict:
    true_labels_dict[dataset_name] = label_encoder.fit_transform(true_labels_dict[dataset_name])

# LDA topic modeling
n_topics_dict = {'Dataset 1': 9, 'Dataset 2': 5, 'Dataset 3': 5}
lda_models = {}

for dataset_name, X_text in vectorized_data.items():
    lda = LatentDirichletAllocation(n_components=n_topics_dict[dataset_name], random_state=42)
    lda.fit(X_text)
    lda_models[dataset_name] = lda

# Model evaluation
for dataset_name, lda in lda_models.items():
    X_text = vectorized_data[dataset_name]
    doc_topic_distribution = lda.transform(X_text)
    predicted_labels = np.argmax(doc_topic_distribution, axis=1)
    true_labels = true_labels_dict[dataset_name]

    # Hungarian algorithm for optimal matching
    cm = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)

    best_predicted_labels = np.zeros_like(predicted_labels)
    for i, j in zip(row_ind, col_ind):
        best_predicted_labels[predicted_labels == j] = i

    # Metrics
    precision = precision_score(true_labels, best_predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, best_predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, best_predicted_labels, average='weighted', zero_division=0)

    print(f"Performance for {dataset_name}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Confusion matrix visualization
    cm_reordered = confusion_matrix(true_labels, best_predicted_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_reordered, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# 3D visualization of clustering results
def plot_3d_clustering(X_3d, labels, title='3D Clustering Visualization'):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels, cmap='Set2', s=45)
    ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.title(title)
    plt.show()


datasets_3d = {
    'Dataset 1': dataset1_reduced,
    'Dataset 2': dataset2_reduced,
    'Dataset 3': dataset3_reduced
}

# Visualizing 3D clustering
for dataset_name, X_numerics in datasets_3d.items():
    labels = true_labels_dict[dataset_name]
    plot_3d_clustering(X_numerics, labels, title=f'3D Clustering for {dataset_name}')
