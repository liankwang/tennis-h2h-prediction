import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.data_processing import create_train_test_sets, create_train_test_sets_elo
from train import train
from evaluate import evaluate, visualize_results, visualize_results_over_cutoffs
import seaborn as sns
import json
import matplotlib.pyplot as plt

def main():
    # Read all ATP matches
    # matches, player_df = read_atp_matches()
    matches = pd.read_csv('data/matches.csv')
    player_df = pd.read_csv('data/expanded_matches.csv')

    cutoff_range = range(2010, 2025)
    accs = []
    aucs = []

    accs_elo = []
    aucs_elo = []

    for cutoff in cutoff_range:
        print(f"Running pipeline for cutoff year: {cutoff}")
        
        output_dir = f'output/logisticregression-{cutoff}'
        os.makedirs(output_dir, exist_ok=True)
        
        output_dir_elo = f'output/logisticregression-elo-{cutoff}'
        os.makedirs(output_dir_elo, exist_ok=True)

        # Create training and testing sets
        # train_data, test_data = create_train_test_sets(matches, player_df, cutoff)
        train_data_elo, test_data_elo = create_train_test_sets_elo('data/matches_with_elo_diff.csv', cutoff)

        # troubleshoot(train_data, test_data, cutoff, output_dir)

        # Train models
        # model = train(train_data, model_type="logistic_regression")
        model_elo = train(train_data_elo, model_type="logistic_regression")

        # Evaluate model and visualize results
        # results = evaluate(model, test_data, model_type="logistic_regression")
        results_elo = evaluate(model_elo, test_data_elo, model_type="logistic_regression")

        # visualize_results(results, cutoff, output_dir)
        # visualize_results(results_elo, cutoff, output_dir_elo)

        # Store accuracy and AUC
        # accs.append(results['acc'])
        # aucs.append(results['auc'])
        accs_elo.append(results_elo['acc'])
        aucs_elo.append(results_elo['auc'])
    
    # visualize_results_over_cutoffs(accs, aucs, cutoff_range, 'output/logisticregression')
    visualize_results_over_cutoffs(accs_elo, aucs_elo, cutoff_range, 'output/logisticregression-elo')
    

def troubleshoot(train_data, test_data, cutoff, output_dir):
    data_info = {
        "train_data_shape": train_data.shape,
        "test_data_shape": test_data.shape,
        "train_label_distribution": train_data['label'].value_counts().to_dict(),
        "test_label_distribution": test_data['label'].value_counts().to_dict(),
        "missing_values_train": train_data.isnull().sum().to_dict(),
        "missing_values_test": test_data.isnull().sum().to_dict()
    }

    # Write the dictionary to a file
    with open(f'{output_dir}/data_info.json', 'w') as f:
        json.dump(data_info, f, indent=4)

    # Check feature correlation
    corr = train_data.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title(f'Feature correlation (cutoff: {cutoff})')
    plt.savefig(f'{output_dir}/feature_correlation.png', dpi=300)
    plt.close()
    # print("\nFeature correlation with label in train data:")
    # print(train_data.corr()['label'].sort_values(ascending=False))



if __name__ == "__main__":
    main()