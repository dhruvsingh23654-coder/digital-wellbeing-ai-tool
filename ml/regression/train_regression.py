from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_regression_model(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

    # Preprocess the data (assuming 'features' and 'target' are defined)
    X = data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature names
    y = data['target']  # Replace with actual target variable name

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    return model

if __name__ == "__main__":
    # Specify the path to the dataset
    dataset_path = '../datasets/processed/your_dataset.csv'  # Update with the actual path
    trained_model = train_regression_model(dataset_path)