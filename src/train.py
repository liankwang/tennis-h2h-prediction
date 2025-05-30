from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd

def train(data, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        print("Training logistic regression model...")
        X = data.drop(columns=['label'])

        X.describe().to_csv('data/feature_summary.csv', index=True)

        y = data['label']

        model = make_pipeline(StandardScaler(), 
                              LogisticRegression(max_iter=1000))
        model.fit(X, y)

        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Only 'logistic_regression' is supported.")

