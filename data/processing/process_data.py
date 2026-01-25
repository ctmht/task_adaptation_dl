import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib



def load_and_preprocess_data(
    csv_path,
    test_size=0.2, # 80/20 split
    val_size=0.1, # 10% of train for val
    random_state=42,
    log_features=('F7', 'F5') #the problematic ones
):
    """
    Loads dataset, applies preprocessing, and returns train/val/test splits.

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_scaler, y_scaler
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Target and features
    X = df.drop(columns='RMSD')
    y = df['RMSD']

    # Log-transform specific skewed features
    X = X.copy()
    for feature in log_features:
        if feature in X.columns:
            X[feature] = np.log1p(X[feature])

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state
    )

    # Standardize only on features 
    # so no data leakage :)
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    joblib.dump(X_scaler, "X_scaler.pkl")

    # scale y separately
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))
    joblib.dump(y_scaler, "y_scaler.pkl")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_scaled, y_val_scaled, y_test_scaled,
        X_scaler, y_scaler
    )
