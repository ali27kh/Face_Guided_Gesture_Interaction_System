import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import pickle
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def train_and_evaluate_models():
    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)

    # Load data
    df = pd.read_csv('./coords.csv')
    X = df.iloc[:, 1:]  # Features (all columns except the first)
    y = df.iloc[:, 0]   # Target value (first column)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    # Define pipelines
    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=1234))    }

    # Train models and compute accuracies
    fit_models = {}
    accuracies = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model
        yhat = model.predict(X_test)
        accuracies[algo] = accuracy_score(y_test, yhat)
        print(f"{algo} Accuracy: {accuracies[algo]}")

    # Check if all accuracies are the same
    same_accuracy = len(set(accuracies.values())) == 1
    selected_model = 'rf' if same_accuracy else None

    # Process outputs for RandomForest only
    model = fit_models['rf']
    yhat = model.predict(X_test)
    report = classification_report(y_test, yhat, output_dict=True)
    
    # Save classification report as text and image
    report_text = f"Classification Report for rf:\n\n" + classification_report(y_test, yhat)
    with open('classification_report_rf.txt', 'w') as f:
        f.write(report_text)
    
    # Create an image of the classification report
    plt.figure(figsize=(10, 6))
    plt.text(0.01, 0.99, report_text, fontsize=12, va='top', fontfamily='monospace')
    plt.axis('off')
    plt.savefig('./plots/classification_report_rf.png', bbox_inches='tight', dpi=100)
    plt.close()

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, yhat)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
    plt.title('Confusion Matrix for rf')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('./plots/confusion_matrix_rf.png', bbox_inches='tight', dpi=100)
    plt.close()

    # Save the RandomForest model
    with open('body_language.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Track loss over time for RandomForest (simulating epochs by increasing trees)
    n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_losses = []
    test_losses = []
    
    for n in n_estimators:
        pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n, random_state=1234))
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities for log loss
        y_train_pred_proba = pipeline.predict_proba(X_train)
        y_test_pred_proba = pipeline.predict_proba(X_test)
        
        # Compute log loss (requires probabilities)
        train_losses.append(log_loss(y_train, y_train_pred_proba))
        test_losses.append(log_loss(y_test, y_test_pred_proba))

    # Plot loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators, train_losses, label='Train Loss')
    plt.plot(n_estimators, test_losses, label='Test Loss')
    plt.xlabel('Number of Trees')
    plt.ylabel('Log Loss')
    plt.title('Loss Over Time for RandomForest')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/loss_over_time_rf.png', bbox_inches='tight', dpi=100)
    plt.close()

    return fit_models

if __name__ == "__main__":
    train_and_evaluate_models()