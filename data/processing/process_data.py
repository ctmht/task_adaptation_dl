import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib



def load_and_preprocess_data(
    csv_path,
    test_size=0.2, # 80/20 split
    random_state=42,
    log_features=('F7', 'F5') #the problematic ones
):
    """
    Loads dataset, applies preprocessing, and returns train/test splits

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Target anf features
    X = df.drop(columns='RMSD')
    y = df['RMSD']

    # Log-transform selected features
    for feature in log_features:
        if feature in X.columns:
            X[feature] = np.log1p(X[feature])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Standardize only on features 
    # so no data leakage :)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # scale y separately
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler, y_scaler
