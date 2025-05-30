from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, test_data, model_type):
    if model_type == 'logistic_regression':
        print("Evaluating logistic regression model...")
        model_only = model.named_steps['logisticregression'] if hasattr(model, 'named_steps') else model

        X_test = test_data.drop(columns=['label'])
        y_true = test_data['label']
        y_probs = model.predict_proba(X_test)[:, 1]

        # ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)

        # Compute predictions using a threshold of 0.5
        y_preds = (y_probs >= 0.5).astype(int)

        # Compute accuracy, precision, recall, and F1
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds)
        recall = recall_score(y_true, y_preds)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_preds)

        # Feature importance
        coefs = pd.Series(model_only.coef_[0], index=X_test.columns)
        coefs.sort_values(ascending=False, inplace=True)

        # Compile results into a dictionary
        results = {
            'y_probs': y_probs,
            'y_preds': y_preds,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': cm,
            'feature_importances': coefs
        }

        return results

def visualize_results(results, cutoff, output_dir):
    # Visualize distribution of probabilities and predictions
    plt.figure(figsize=(8, 6))
    plt.hist(results['y_probs'], bins=50, color='tab:blue', label='Predicted probabilities')
    plt.axvline(x=0.5, color='tab:orange', linestyle='--', label='Threshold (0.5)')
    plt.title(f'Distribution of predicted probabilities (cutoff: {cutoff})')
    plt.xlabel('Predicted probability')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/probs_distribution_cutoff_{cutoff}.png', dpi=300)
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr'], results['tpr'], color='tab:blue', label=f'AUC = {results["auc"]:.4f}')
    plt.plot([0, 1], [0, 1], color='tab:orange', linestyle='--', label='Random guess')
    plt.title(f'ROC (cutoff: {cutoff})')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve_cutoff_{cutoff}.png', dpi=300)
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(results['confusion_matrix'], cmap='Blues', interpolation='nearest')
    plt.title(f'Confusion matrix (cutoff: {cutoff})')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(ticks=[0, 1], labels=['Loser', 'Winner'])
    plt.yticks(ticks=[0, 1], labels=['Loser', 'Winner'])
    plt.savefig(f'{output_dir}/confusion_matrix_cutoff_{cutoff}.png', dpi=300)
    plt.close()

    # Plot feature importances
    coefs = results['feature_importances']
    coefs.to_csv(f'{output_dir}/feature_importances.csv', header=True)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=results['feature_importances'].values, y=results['feature_importances'].index, palette='viridis', orient='h')
    plt.title(f'Feature importances (cutoff: {cutoff})')
    plt.xlabel('Standardized coefficient value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importances_cutoff_{cutoff}.png', dpi=300)
    plt.close()

    # Write summary metrics to a results file
    summary_file = f'{output_dir}/results_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Accuracy: {results['acc']:.4f}\n")
        f.write(f"AUC: {results['auc']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")

def visualize_results_over_cutoffs(accs, aucs, cutoff_range, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(cutoff_range, accs, marker='o', label='Accuracy')
    plt.plot(cutoff_range, aucs, marker='x', label='AUC')
    plt.title('Model performance across cutoff years')
    plt.xlabel('Cutoff years')
    plt.ylabel('Score')
    plt.xticks(cutoff_range)
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/performance_across_cutoffs.png', dpi=300)
    plt.close()
