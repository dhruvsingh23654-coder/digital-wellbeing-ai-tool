from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Load the dataset from the specified file path."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data for clustering."""
    # Example preprocessing: drop non-numeric columns and handle missing values
    data = data.select_dtypes(include=[np.number]).dropna()
    return data

def train_kmeans(data, n_clusters):
    """Train K-Means clustering model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

def save_model(model, model_path):
    """Save the trained model to a file."""
    import joblib
    joblib.dump(model, model_path)

def main():
    # Define file paths
    raw_data_path = os.path.join('ml', 'datasets', 'raw', 'smartphone_usage.csv')  # Example file name
    model_path = os.path.join('ml', 'models', 'kmeans_model.pkl')  # Adjust path as necessary

    # Load and preprocess data
    data = load_data(raw_data_path)
    processed_data = preprocess_data(data)

    # Train K-Means model
    n_clusters = 5  # Define the number of clusters
    kmeans_model = train_kmeans(processed_data, n_clusters)

    # Save the trained model
    save_model(kmeans_model, model_path)

if __name__ == "__main__":
    main()