from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate(model, test_data, model_type):
    if model_type == 'logistic_regression':
        print("Evaluating logistic regression model...")
        X_test = test_data.drop(columns=['label'])
        y_true = test_data['label']
        y_pred = model.predict(X_test)
        print("Predicted y distribution: ", pd.Series(y_pred).value_counts())

        acc = accuracy_score(y_true, y_pred)
        print(classification_report(y_true, y_pred))

        return acc