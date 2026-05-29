# MindSync AI

MindSync AI is an academic-focused web application designed to analyze user lifestyle and smartphone usage patterns. By leveraging advanced AI and machine learning techniques, MindSync AI predicts key metrics such as Digital Wellbeing Score, Smartphone Addiction Risk, and Productivity Level.

## Features

- **Digital Wellbeing Score Prediction**: Utilizes regression models to assess and predict the user's digital wellbeing based on their smartphone usage patterns.
- **Smartphone Addiction Risk Assessment**: Implements classification algorithms to evaluate the risk of smartphone addiction, providing users with insights into their usage habits.
- **Productivity Level Analysis**: Analyzes user data to predict productivity levels, helping users optimize their time and improve efficiency.
- **Data Visualization Dashboard**: Offers an interactive dashboard for users to visualize their data and predictions, enhancing user engagement and understanding.

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
MindSync-AI
├── app
│   ├── __init__.py
│   ├── routes
│   │   ├── __init__.py
│   │   └── main.py
│   ├── templates
│   │   ├── index.html
│   │   ├── results.html
│   │   └── dashboard.html
│   ├── static
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── main.js
│   └── services
│       └── predictor.py
├── ml
│   ├── datasets
│   │   ├── raw
│   │   └── processed
│   ├── notebooks
│   │   └── exploration.ipynb
│   ├── preprocessing
│   │   └── preprocess.py
│   ├── regression
│   │   └── train_regression.py
│   ├── classification
│   │   └── train_classification.py
│   ├── clustering
│   │   └── train_clustering.py
│   └── evaluation
│       └── metrics.py
├── tests
│   ├── __init__.py
│   └── test_app.py
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MindSync-AI.git
   cd MindSync-AI
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5000`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.