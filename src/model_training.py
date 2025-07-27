import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, preprocess_data, prepare_data

def train_models(X_train, y_train):
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    # XGBoost
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    return lr, xgb

def evaluate_models(model, X_test, y_test, model_name, dataset_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    
    # F1-Score
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} on {dataset_name}:")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'reports/{model_name}_{dataset_name}_cm.png')
    plt.close()
    
    return auc_pr, f1

def main():
    # Load preprocessed data (from data_preprocessing.py)
    fraud_data, ip_data, creditcard_data = load_data('../data/raw/Fraud_Data.csv', '../data/raw/IpAddress_to_Country.csv', '../data/raw/creditcard.csv')
    fraud_data, creditcard_data = preprocess_data(fraud_data, ip_data, creditcard_data)
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test = prepare_data(fraud_data, creditcard_data)
    
    # Train models
    lr_fraud, xgb_fraud = train_models(X_fraud_train, y_fraud_train)
    lr_credit, xgb_credit = train_models(X_credit_train, y_credit_train)
    
    # Evaluate models
    lr_fraud_metrics = evaluate_models(lr_fraud, X_fraud_test, y_fraud_test, 'LogisticRegression', 'Fraud_Data')
    xgb_fraud_metrics = evaluate_models(xgb_fraud, X_fraud_test, y_fraud_test, 'XGBoost', 'Fraud_Data')
    lr_credit_metrics = evaluate_models(lr_credit, X_credit_test, y_credit_test, 'LogisticRegression', 'creditcard')
    xgb_credit_metrics = evaluate_models(xgb_credit, X_credit_test, y_credit_test, 'XGBoost', 'creditcard')
    
    # Select best model
    best_model = 'XGBoost' if xgb_fraud_metrics[0] > lr_fraud_metrics[0] and xgb_credit_metrics[0] > lr_credit_metrics[0] else 'LogisticRegression'
    print(f"Best model: {best_model} based on AUC-PR.")

if __name__ == "__main__":
    main()