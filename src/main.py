import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.data_processing import create_train_test_sets
from train import train
from evaluate import evaluate

def main():
    # Read all ATP matches
    # matches, player_df = read_atp_matches()
    matches = pd.read_csv('data/matches.csv')
    player_df = pd.read_csv('data/expanded_matches.csv')

    for cutoff in range(2024, 2025):
        # Create training and testing sets
        train_data, test_data = create_train_test_sets(matches, player_df, cutoff)

        troubleshoot(train_data, test_data)

        # Train model
        model = train(train_data, model_type="logistic_regression")

        # Evaluate model
        acc = evaluate(model, test_data, model_type="logistic_regression")
        print(f"Accuracy for cutoff year {cutoff}: {acc:.4f}")

def troubleshoot(train_data, test_data):
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    # Check label distribution
    print("Train data label distribution:")
    print(train_data['label'].value_counts())

    print("Test data label distribution:")
    print(test_data['label'].value_counts())

    # Check for missing values
    print("\nMissing values in train data:")
    print(train_data.isnull().sum())

    print("\nMissing values in test data:")
    print(test_data.isnull().sum())

    # Check correlation between features and target
    print("\nFeature correlation with label in train data:")
    print(train_data.corr()['label'].sort_values(ascending=False))


if __name__ == "__main__":
    main()