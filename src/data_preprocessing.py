import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime

def load_data(fraud_path, ip_path, creditcard_path):
    try:
        fraud_data = pd.read_csv(fraud_path)
        ip_data = pd.read_csv(ip_path, encoding='utf-8-sig')
        creditcard_data = pd.read_csv(creditcard_path)
        expected_cols = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
        if not all(col in ip_data.columns for col in expected_cols):
            print(f"IP data columns: {ip_data.columns.tolist()}")
            raise ValueError("Expected columns in IpAddress_to_Country.csv: " + ", ".join(expected_cols))
        return fraud_data, ip_data, creditcard_data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def handle_missing_values(df):
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def clean_data(df):
    df = df.drop_duplicates()
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def merge_ip_data(fraud_data, ip_data):
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    # Sort IP data for efficient lookup
    ip_data = ip_data.sort_values('lower_bound_ip_address')
    lower_bounds = ip_data['lower_bound_ip_address'].values
    upper_bounds = ip_data['upper_bound_ip_address'].values
    countries = ip_data['country'].values

    def map_ip_to_country(ip):
        # Find the first index where lower_bound <= ip
        idx = np.searchsorted(lower_bounds, ip, side='right') - 1
        if idx >= 0 and lower_bounds[idx] <= ip <= upper_bounds[idx]:
            return countries[idx]
        return 'Unknown'

    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    return fraud_data

def feature_engineering(fraud_data):
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['avg_time_between_transactions'] = fraud_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0) / 3600
    return fraud_data

def preprocess_data(fraud_data, ip_data, creditcard_data):
    fraud_data = handle_missing_values(fraud_data)
    creditcard_data = handle_missing_values(creditcard_data)
    fraud_data = clean_data(fraud_data)
    creditcard_data = clean_data(creditcard_data)
    fraud_data = merge_ip_data(fraud_data, ip_data)
    fraud_data = feature_engineering(fraud_data)
    categorical_cols = ['source', 'browser', 'sex', 'country']
    fraud_data = pd.get_dummies(fraud_data, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    numerical_cols_fraud = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'avg_time_between_transactions']
    fraud_data[numerical_cols_fraud] = scaler.fit_transform(fraud_data[numerical_cols_fraud])
    numerical_cols_credit = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    creditcard_data[numerical_cols_credit] = scaler.fit_transform(creditcard_data[numerical_cols_credit])
    return fraud_data, creditcard_data

def handle_class_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def prepare_data(fraud_data, creditcard_data):
    try:
        X_fraud = fraud_data.drop(['class', 'user_id', 'device_id', 'signup_time', 'purchase_time'], axis=1)
        y_fraud = fraud_data['class']
        X_credit = creditcard_data.drop(['Class'], axis=1)
        y_credit = creditcard_data['Class']
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
        X_fraud_train, y_fraud_train = handle_class_imbalance(X_fraud_train, y_fraud_train)
        X_credit_train, y_credit_train = handle_class_imbalance(X_credit_train, y_credit_train)
        return X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise

if __name__ == "__main__":
    try:
        fraud_data, ip_data, creditcard_data = load_data('../data/raw/Fraud_Data.csv', '../data/raw/IpAddress_to_Country.csv', '../data/raw/creditcard.csv')
        fraud_data, creditcard_data = preprocess_data(fraud_data, ip_data, creditcard_data)
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test = prepare_data(fraud_data, creditcard_data)
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")