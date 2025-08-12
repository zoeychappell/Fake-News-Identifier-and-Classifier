'''
Zoey Chappell, Luke McEwen, and Daniel Wolosiuk
Saniat Sohrawardi
CSEC 520

AI Usage Statement
Tools Used: ChatGPT
- Usage: Brainstorming suitable libraries. Error detection. Regex format. 
- Verification: Cross-checked with library manual page and manual testing
Prohibited Use Compliance: Confirmed
'''

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
import data_process as dp
import matplotlib.pyplot as plt
from collections import defaultdict

def save_clusters(clusters, cleaned_text):
    """
    Groups cleaned text data by their cluster labels and appends a summary to 'texts.csv'.

    Parameters:
        clusters (list or array-like): A list of cluster labels corresponding to each text sample.
        cleaned_text (list of str): A list of preprocessed text samples.

    Notes:
        - The output is appended to 'texts.csv'.
        - Only the first 100 text samples per cluster are written.
        - Each entry in the file includes the cluster ID and the number of samples.
    """
    cluster_groups = defaultdict(list)
    for i, cluster in enumerate(clusters):
        cluster_groups[cluster].append(cleaned_text[i])
    with open("texts.csv", "a") as file:
        file.write("\n=== Cluster Summaries ===")
        for cluster_id, texts in cluster_groups.items():
            file.write(f"\nCluster {cluster_id}: ({len(texts)} samples)")
            for text in texts[:100]:  # Show first 100 texts
                file.write(f"  - {text}...")

def evaluate_kmeans_range(X, true_labels, max_k=10):
    """
    Evaluates KMeans clustering performance over a range of cluster counts and returns metrics.

    Parameters:
        X (array-like): Feature matrix used for clustering.
        true_labels (array-like): Ground truth class labels for evaluation.
        max_k (int, optional): Maximum number of clusters to test. Defaults to 10.

    Returns:
        dict: A dictionary containing lists of evaluation metrics for each cluster count `k`, including:
              - 'k': tested number of clusters
              - 'accuracy': clustering accuracy after label alignment
              - 'precision': macro-averaged precision
              - 'recall': macro-averaged recall
              - 'f1': macro-averaged F1-score

    Notes:
        - For `k=2`, the function uses a label-flipping trick to align clusters with true labels.
        - Metrics assume best-effort alignment between predicted and true labels.
    """
    results = {
        'k': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        predicted = kmeans.fit_predict(X)

        # Match predicted labels to true labels as best as possible
        # If k == 2, use flipping trick
        if k == 2:
            flipped = 1 - predicted
            acc1 = accuracy_score(true_labels, predicted)
            acc2 = accuracy_score(true_labels, flipped)
            if acc2 > acc1:
                predicted = flipped

        results['k'].append(k)
        results['accuracy'].append(accuracy_score(true_labels, predicted))
        results['precision'].append(precision_score(true_labels, predicted, average='macro'))
        results['recall'].append(recall_score(true_labels, predicted, average='macro'))
        results['f1'].append(f1_score(true_labels, predicted, average='macro'))

    return results

def plot_kmeans_metrics(results):
    """
    Plots evaluation metrics of KMeans clustering over different cluster counts.

    Parameters:
        results (dict): Dictionary containing lists of metrics returned by `evaluate_kmeans_range`.
                        Keys should include 'k', 'accuracy', 'precision', 'recall', and 'f1'.

    Displays:
        A line plot showing how each metric varies with the number of clusters.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['k'], results['accuracy'], marker='o', label='Accuracy')
    plt.plot(results['k'], results['precision'], marker='o', label='Precision')
    plt.plot(results['k'], results['recall'], marker='o', label='Recall')
    plt.plot(results['k'], results['f1'], marker='o', label='F1 Score')
    
    plt.title('KMeans Performance vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score')
    plt.xticks(results['k'])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_dbscan_range(X, true_labels, eps_values, min_samples=5):
    """
    Evaluates DBSCAN clustering over a range of epsilon values and returns performance metrics.

    Parameters:
        X (array-like): Feature matrix for clustering.
        true_labels (array-like): Ground truth labels for evaluation.
        eps_values (list of float): List of `eps` values to evaluate in DBSCAN.
        min_samples (int, optional): Minimum number of samples for a core point in DBSCAN. Defaults to 5.

    Returns:
        dict: A dictionary containing:
            - 'eps': List of epsilon values used.
            - 'accuracy': Accuracy scores (excluding noise).
            - 'precision': Macro-averaged precision (excluding noise).
            - 'recall': Macro-averaged recall (excluding noise).
            - 'f1': Macro-averaged F1 scores (excluding noise).
            - 'n_clusters': Number of clusters found (excluding noise).
            - 'n_noise': Number of noise points (label -1).

    Notes:
        - Noise points are excluded from metric calculations.
        - Metrics are only computed for non-noise data.
        - Results are printed for each epsilon for debugging purposes.
    """
    results = {
        'eps': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'n_clusters': [],
        'n_noise': []
    }

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        predicted = db.fit_predict(X)

        # Filter out noise (-1)
        mask = predicted != -1
        filtered_true = np.array(true_labels)[mask]
        filtered_pred = np.array(predicted)[mask]

        #if len(set(filtered_pred)) <= 1:
            # Skip if only one cluster or all noise
        #    continue

        results['eps'].append(eps)
        results['accuracy'].append(accuracy_score(filtered_true, filtered_pred))
        results['precision'].append(precision_score(filtered_true, filtered_pred, average='macro'))
        results['recall'].append(recall_score(filtered_true, filtered_pred, average='macro'))
        results['f1'].append(f1_score(filtered_true, filtered_pred, average='macro'))
        results['n_clusters'].append(len(set(filtered_pred)))
        results['n_noise'].append(list(predicted).count(-1))

    return results


def plot_dbscan_metrics(results):
    """
    Plots evaluation metrics of DBSCAN clustering across different epsilon (eps) values.

    Parameters:
        results (dict): Dictionary containing DBSCAN evaluation results from `evaluate_dbscan_range`.
                        Must include keys: 'eps', 'accuracy', 'precision', 'recall', and 'f1'.

    Displays:
        A line plot showing how clustering metrics vary with the epsilon parameter.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['eps'], results['accuracy'], marker='o', label='Accuracy')
    plt.plot(results['eps'], results['precision'], marker='o', label='Precision')
    plt.plot(results['eps'], results['recall'], marker='o', label='Recall')
    plt.plot(results['eps'], results['f1'], marker='o', label='F1 Score')
    
    plt.title('DBSCAN Performance vs Epsilon (eps)')
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Score')
    plt.xticks(results['eps'])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_dbscan_clusters_to_csv(cleaned_texts, predicted, filename="dbscan_clusters.csv"):
    """
    Saves DBSCAN clustering results to a CSV file with cluster labels and corresponding texts.

    Parameters:
        cleaned_texts (list of str): List of cleaned/preprocessed text samples.
        predicted (list or array-like): Cluster labels assigned by DBSCAN (including -1 for noise).
        filename (str, optional): Output CSV filename. Defaults to 'dbscan_clusters.csv'.

    Output:
        A CSV file where each row contains a cluster label and its corresponding text.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Cluster', 'Text'])
        for text, label in zip(cleaned_texts, predicted):
            writer.writerow([label, text])

def main():
    messages, label = dp.tokenize("./fake_and_real_news.csv")
    cleaned_messages = dp.clean(messages)

    # Flatten tokens into strings for TF-IDF
    cleaned_text = [" ".join(msg) for msg in cleaned_messages]

    # TF-IDF feature extraction
    tfidf_matrix, vectorizer = dp.tf_idf(cleaned_text, max_feature=500)

    # Optional: reduce dimensionality
    X_pca, pca_model = dp.pca(tfidf_matrix, n_components=50)

    # Encode the 'Fake' and 'Real' labels as 0 and 1
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(label)

    results = evaluate_kmeans_range(X_pca, true_labels, max_k=20)
    print(f"K-MEANS STATS{results}")
    #plot_kmeans_metrics(results)

    # Try a range of eps values (e.g., from 0.3 to 1.2)
    eps_range = [round(eps, 2) for eps in np.arange(0.3, 1.3, 0.1)] # took best which is esp=0.5
    results = evaluate_dbscan_range(X_pca, true_labels, eps_range, min_samples=5)
    print(f"DBSCAN STATS\n{results}")

    # Plot the result
    #plot_dbscan_metrics(results)

    #if predicted is not None:
    #    save_dbscan_clusters_to_csv(cleaned_text, predicted)

    # Group texts by cluster
    #save_clusters(clusters, cleaned_text)

if __name__ == "__main__":
    main()
