from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def train_classification_model(data_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Preprocess data (assuming 'features' and 'target' are defined)
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classifiers
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # Train models and evaluate
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Print evaluation metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_classification_model('path/to/your/dataset.csv')  # Update with the actual dataset path