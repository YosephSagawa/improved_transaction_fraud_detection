import shap
import xgboost
import matplotlib.pyplot as plt
from data_preprocessing import load_data, preprocess_data, prepare_data
from xgboost import XGBClassifier
def shap_analysis(model, X_train, X_test, dataset_name):
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'SHAP Summary Plot - {dataset_name}')
    plt.savefig(f'reports/shap_summary_{dataset_name}.png')
    plt.close()
    
    # Force Plot for first sample
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False, matplotlib=True)
    plt.savefig(f'reports/shap_force_{dataset_name}.png')
    plt.close()

def main():
    # Load preprocessed data
    fraud_data, ip_data, creditcard_data = load_data('../data/raw/Fraud_Data.csv', '../data/raw/IpAddress_to_Country.csv', '../data/raw/creditcard.csv')
    fraud_data, creditcard_data = preprocess_data(fraud_data, ip_data, creditcard_data)
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test = prepare_data(fraud_data, creditcard_data)
    
    # Train XGBoost
    xgb_fraud = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_fraud.fit(X_fraud_train, y_fraud_train)
    xgb_credit = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_credit.fit(X_credit_train, y_credit_train)
    
    # SHAP analysis
    shap_analysis(xgb_fraud, X_fraud_train, X_fraud_test, 'Fraud_Data')
    shap_analysis(xgb_credit, X_credit_train, X_credit_test, 'creditcard')

if __name__ == "__main__":
    main()