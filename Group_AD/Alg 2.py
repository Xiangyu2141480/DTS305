import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings("ignore")


def evaluate_clustering(X, y, n_components_range, final_n_components, random_state=42):
    results = {'n_components': [], 'precision': [], 'recall': [], 'f1_score': []}

    for n_components in n_components_range:
        lsa = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_lsa = lsa.fit_transform(X)

        clusterer_lsa = AgglomerativeClustering(n_clusters=len(set(y)), metric='euclidean', linkage='ward')
        cluster_labels_lsa = clusterer_lsa.fit_predict(X_lsa)

        conf_matrix = confusion_matrix(y, cluster_labels_lsa)
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        matched_labels = np.zeros_like(cluster_labels_lsa)
        for i, j in zip(row_ind, col_ind):
            matched_labels[cluster_labels_lsa == j] = i

        precision = precision_score(y, matched_labels, average='macro')
        recall = recall_score(y, matched_labels, average='macro')
        f1 = f1_score(y, matched_labels, average='macro')

        results['n_components'].append(n_components)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)

    results_df = pd.DataFrame(results)

    results_df.plot(x='n_components', y=['precision', 'recall', 'f1_score'], marker='o', figsize=(10, 6))

    plt.title('Model Performance with Different n_components', fontsize=14)
    plt.xlabel('n_components', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True)
    plt.show()

    # Final clustering with specified number of components
    lsa = TruncatedSVD(n_components=final_n_components, random_state=random_state)
    X_lsa = lsa.fit_transform(X)

    clusterer_lsa = AgglomerativeClustering(n_clusters=len(set(y)), metric='euclidean', linkage='ward')
    cluster_labels_lsa = clusterer_lsa.fit_predict(X_lsa)

    conf_matrix = confusion_matrix(y, cluster_labels_lsa)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    matched_labels = np.zeros_like(cluster_labels_lsa)
    for i, j in zip(row_ind, col_ind):
        matched_labels[cluster_labels_lsa == j] = i

    precision = precision_score(y, matched_labels, average='macro')
    recall = recall_score(y, matched_labels, average='macro')
    f1 = f1_score(y, matched_labels, average='macro')
    silhouette_avg = silhouette_score(X_lsa, matched_labels)
    davies_bouldin = davies_bouldin_score(X_lsa, matched_labels)
    calinski_harabasz = calinski_harabasz_score(X_lsa, matched_labels)

    conf_matrix = confusion_matrix(y, matched_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix for n_components = {final_n_components}', fontsize=14)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.show()

    # 3D visualization even if data is 2D
    if final_n_components == 2:
        zeros_column = np.zeros((X_lsa.shape[0], 1))
        X_lsa_3d = np.hstack((X_lsa, zeros_column))
    else:
        X_lsa_3d = X_lsa

    if X_lsa_3d.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_lsa_3d[:, 0], X_lsa_3d[:, 1], X_lsa_3d[:, 2], c=matched_labels, cmap='viridis', s=50)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_title('3D Visualization of Clusters', fontsize=14)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.show()

    print(f"For n_components = {final_n_components}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz}")


# Step 1: Load the dataset
file_path = 'Data Set 1.csv'
articles_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Check for missing values (NaN) in the dataset
missing_values = articles_data.isnull().sum()

# Check for duplicate rows in the dataset
duplicate_rows = articles_data.duplicated().sum()

print(missing_values, duplicate_rows)

# Step 2: Preprocess the data
articles_data = articles_data.drop_duplicates()
# Fill any missing values in the 'Article' column with an empty string
articles_text = articles_data['Article'].fillna('')

# Step 3: Encode the 'NewsType' column into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(articles_data['NewsType'].fillna(''))

# Step 4: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
X = tfidf_vectorizer.fit_transform(articles_text)

evaluate_clustering(X, y, n_components_range=range(2, 11), final_n_components=8)

# # Dataset 2

file_path = 'Data Set 2.csv'
articles_data = pd.read_csv(file_path)

# Check for missing values (NaN) in the dataset
missing_values = articles_data.isnull().sum()

# Check for duplicate rows in the dataset
duplicate_rows = articles_data.duplicated().sum()

print(missing_values, duplicate_rows)

# Step 2: Preprocess the data
articles_data = articles_data.drop_duplicates()
# Fill any missing values in the 'text' column with an empty string
articles_text = articles_data['text'].fillna('')

# Step 3: Encode the 'label' column into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(articles_data['label'].fillna(''))

# Step 4: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
X = tfidf_vectorizer.fit_transform(articles_text)

evaluate_clustering(X, y, n_components_range=range(2, 11), final_n_components=4)

# # Dataset 3

file_path = 'Data Set 3.csv'
articles_data = pd.read_csv(file_path)

# Check for missing values (NaN) in the dataset
missing_values = articles_data.isnull().sum()

# Check for duplicate rows in the dataset
duplicate_rows = articles_data.duplicated().sum()

print(missing_values, duplicate_rows)

# Step 2: Preprocess the data
articles_data = articles_data.drop_duplicates()
# Fill any missing values in the 'Article' column with an empty string
articles_text = articles_data['News'].fillna('')

# Step 3: Encode the 'NewsType' column into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(articles_data['Price Sentiment'].fillna(''))

# Step 4: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
X = tfidf_vectorizer.fit_transform(articles_text)

evaluate_clustering(X, y, n_components_range=range(2, 11), final_n_components=3)
