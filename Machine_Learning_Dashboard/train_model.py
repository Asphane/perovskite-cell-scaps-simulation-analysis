import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def main():
    csv_path = 'Machine_Learning_Dashboard/ml_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run extract_data.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Define targets
    targets = ['PCE', 'Voc', 'Jsc', 'FF']
    
    # Define features (everything else except Source_File and Targets)
    features = [col for col in df.columns if col not in targets and col != 'Source_File']
    
    print(f"Features ({len(features)}):", features)
    print(f"Targets ({len(targets)}):", targets)
    
    X = df[features]
    y = df[targets]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with an Imputer (to handle NaNs for parameters not swept in some files)
    # and a Random Forest Regressor
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("\nTraining Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # Convert predictions to dataframe for easier metric calculation
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    
    for i, target in enumerate(targets):
        r2 = r2_score(y_test.iloc[:, i], y_pred_df.iloc[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_df.iloc[:, i]))
        print(f"Target: {target:4} | R^2 Score: {r2:.4f} | RMSE: {rmse:.4f}")
        
    # Save the model
    model_path = 'Machine_Learning_Dashboard/psc_surrogate_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"\nModel successfully saved to {model_path}")
    
    # Save feature names so we can use them in the Streamlit app
    feature_names_path = 'Machine_Learning_Dashboard/feature_names.pkl'
    joblib.dump(features, feature_names_path)
    
if __name__ == "__main__":
    main()
