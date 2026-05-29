from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Convert categorical variables to numerical
    data = pd.get_dummies(data, drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    return pd.DataFrame(scaled_features, columns=data.columns)

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_processed_data(data, output_path):
    data.to_csv(output_path, index=False)