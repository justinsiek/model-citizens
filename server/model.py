import gc
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Configuration
class config:
    seed = 42
    n_splits = 10

# Load data from cleandata if available, otherwise process it
def load_data():
    # Try to load processed data first
    if os.path.exists('processed_data.csv'):
        print("Loading preprocessed data...")
        data = pd.read_csv('processed_data.csv')
    else:
        # If not available, import from cleandata.py
        print("Preprocessed data not found. Running data preparation...")
        from cleandata import prepare_data
        data = prepare_data()
    
    return data

# Process a single string (used for test data)
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join(sentences)

def main():
    # Load training data
    print("Loading data...")
    train = load_data()
    
    # Load test data if available
    if os.path.exists('test.csv'):
        test = pd.read_csv('test.csv')
        
        # Process test data
        test["prompt"] = test["prompt"].apply(process)
        test["response_a"] = test["response_a"].apply(process)
        test["response_b"] = test["response_b"].apply(process)
        
        # Extract features for test data
        from cleandata import Preprocessor
        preprocessor = Preprocessor()
        test = preprocessor.run(test)
    else:
        print("Test file not found. Will skip prediction.")
        test = None
    
    # Check if we need to create sample submission
    sample_submission = None
    if test is not None and os.path.exists('sample_submission.csv'):
        sample_submission = pd.read_csv('sample_submission.csv')
    elif test is not None:
        # Create a sample submission format
        sample_submission = pd.DataFrame({
            "id": test["id"] if "id" in test.columns else list(range(len(test))),
            "winner_model_a": [0] * len(test),
            "winner_model_b": [0] * len(test),
            "winner_tie": [0] * len(test)
        })
    
    print(f"Train shape: {train.shape}")
    if test is not None:
        print(f"Test shape: {test.shape}")
    print("-"*90)
    print(f"Train missing values: {train.isnull().sum().sum()}")
    if test is not None:
        print(f"Test missing values: {test.isnull().sum().sum()}")
    print("-"*90)
    
    # Define columns to drop - only drop columns that exist
    base_cols_to_drop = ["prompt", "response_a", "response_b"]
    drop_cols = [col for col in base_cols_to_drop if col in train.columns]
    
    if "id" in train.columns:
        drop_cols.append("id")
    
    # Only include target columns that exist in the dataframe
    all_target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    target_cols = [col for col in all_target_cols if col in train.columns]
    target = "target"
    
    # Create target variable if not already present
    if target not in train.columns:
        train[target] = np.nan
        for idx, t in enumerate(target_cols):
            if t in train.columns:
                train.loc[train[t] == 1, target] = idx
        train[target] = train[target].astype("int32")
    
    # Prepare data for modeling - only drop columns that exist
    columns_to_drop = []
    
    # Add target columns that exist
    columns_to_drop.extend([col for col in target_cols if col in train.columns])
    
    # Add base columns to drop
    columns_to_drop.extend(drop_cols)
    
    # Add target column
    if target in train.columns:
        columns_to_drop.append(target)
    
    # Add model columns if they exist
    if "model_a" in train.columns:
        columns_to_drop.append("model_a")
    if "model_b" in train.columns:
        columns_to_drop.append("model_b")
    
    # Drop columns
    X = train.drop(columns=columns_to_drop, errors='ignore')
    y = train[target]
    
    # Handle test data
    if test is not None:
        test_drop_cols = [col for col in drop_cols if col in test.columns]
        X_test = test.drop(columns=test_drop_cols, errors='ignore')
    
    # Replace infinities with NaN
    X = X.replace([-np.inf, np.inf], np.nan)
    X = X.fillna(0)  # Fill NaNs with 0
    
    if test is not None:
        X_test = X_test.replace([-np.inf, np.inf], np.nan)
        X_test = X_test.fillna(0)  # Fill NaNs with 0
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    
    # Create containers for predictions and metrics
    if test is not None:
        test_preds = np.zeros(shape=(X_test.shape[0], y.nunique()))
    cv_scores = list()
    accuracy_scores = list()  # To track accuracy across folds
    
    # Get unique classes for proper model setup
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"Number of unique classes in data: {num_classes} {unique_classes}")
    
    # Ensure we're only doing binary classification
    if num_classes != 2:
        print("Warning: This model is designed for binary classification (model A wins vs model B wins)")
        print("The data should only contain wins (0) and losses (1), with no ties")
    
    # Train models with cross-validation
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"| Fold {idx+1} |".center(90, "="))
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        
        print(f'Train: {X_train.shape}')
        print(f'Val: {X_val.shape}')
        
        # Initialize model for binary classification
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Changed to binary classification
            eval_metric='logloss',
            subsample=0.8,
            n_estimators=1000,  # Increased for better performance
            learning_rate=0.03,  # Reduced for better generalization
            max_depth=6,         # Slightly increased
            colsample_bytree=0.8, # Added for better feature selection
            min_child_weight=3,   # Added to prevent overfitting
            reg_lambda=2.0,       # L2 regularization to prevent overfitting
            random_state=config.seed,
            early_stopping_rounds=75
        )
        
        # Train model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=75
        )
        
        # Evaluate model
        val_preds = model.predict_proba(X_val)
        val_pred_labels = np.argmax(val_preds, axis=1)
        val_accuracy = np.mean(val_pred_labels == y_val)
        val_log_loss_score = log_loss(y_val, val_preds, labels=unique_classes)
        
        print(f"Val log loss: {val_log_loss_score:.5f}")
        print(f"Val accuracy: {val_accuracy:.5f} ({np.sum(val_pred_labels == y_val)}/{len(y_val)})")
        
        cv_scores.append(val_log_loss_score)
        accuracy_scores.append(val_accuracy)
        
        # Predict on test data
        if test is not None:
            test_preds += model.predict_proba(X_test) / cv.get_n_splits()
        
        # Save model
        model.save_model(f"xgb_model_fold_{idx+1}.json")
        
        # Clean up to free memory
        del model
        gc.collect()
    
    # Print cross-validation result
    print("="*90)
    print(f"CV Log Loss: {np.mean(cv_scores):.5f}")
    print(f"CV Accuracy: {np.mean(accuracy_scores):.5f} (Average across all folds)")
    
    # Save predictions to submission file
    if test is not None and sample_submission is not None:
        for idx, t in enumerate(target_cols):
            sample_submission[t] = test_preds[:, idx]
        sample_submission.to_csv("submission.csv", index=False)
        print("Predictions saved to submission.csv")
    
    print("Done!")

def predict_preference(prompt, response_a, response_b):
    """
    Make a prediction for a single example using the trained model.
    Returns probability of preferring model A or model B (binary).
    """
    # Load the model from fold 1 (could average multiple folds)
    model = xgb.XGBClassifier()
    model.load_model("xgb_model_fold_1.json")
    
    # Process the data
    from cleandata import Preprocessor
    
    # Process inputs
    processed_prompt = process(prompt)
    processed_response_a = process(response_a)
    processed_response_b = process(response_b)
    
    # Create dataframe for feature extraction
    test_data = pd.DataFrame({
        'prompt': [processed_prompt],
        'response_a': [processed_response_a],
        'response_b': [processed_response_b],
        'winner_model_b': [0]  # Placeholder, not used for prediction
    })
    
    # Extract features
    preprocessor = Preprocessor()
    processed_data = preprocessor.run(test_data)
    
    # Prepare for prediction
    drop_cols = ["prompt", "response_a", "response_b", "winner_model_b"]
    X = processed_data.drop(columns=drop_cols, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Make prediction - binary classification (model A vs model B)
    probabilities = model.predict_proba(X)[0]
    
    return {
        "prefer_a_probability": 1 - probabilities[1],  # Probability of model A winning
        "prefer_b_probability": probabilities[1]       # Probability of model B winning
    }

if __name__ == "__main__":
    main()