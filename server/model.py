import gc
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.utils import resample

# Configuration
class config:
    seed = 42
    validation_size = 0.2  # Size of validation set
    n_model_variations = 50  # Number of model variations to train
    # Data sampling settings
    min_sample_fraction = 0.3  # Minimum fraction of samples to use
    max_sample_fraction = 1.0  # Maximum fraction of samples to use
    min_feature_fraction = 0.3  # Minimum fraction of features to use
    max_feature_fraction = 1.0  # Maximum fraction of features to use

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

# Generate XGBoost model variations with different hyperparameters
def generate_model_variations(n_variations=50, seed=42):
    np.random.seed(seed)
    model_variations = []
    
    for i in range(n_variations):
        # Use different random seeds for each model
        model_seed = seed + i
        
        # Create hyperparameter variations
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': np.random.randint(500, 1500),
            'learning_rate': np.random.uniform(0.01, 0.1),
            'max_depth': np.random.randint(3, 9),
            'subsample': np.random.uniform(0.6, 1.0),
            'colsample_bytree': np.random.uniform(0.6, 1.0),
            'min_child_weight': np.random.randint(1, 6),
            'reg_lambda': np.random.uniform(0.1, 5.0),
            'reg_alpha': np.random.uniform(0, 2.0),
            'random_state': model_seed,
            'early_stopping_rounds': 75
        }
        
        # For some models, use more specialized parameters
        if np.random.random() < 0.2:
            # Add some specialized variations
            params['gamma'] = np.random.uniform(0, 5)
            params['grow_policy'] = np.random.choice(['depthwise', 'lossguide'])
        
        # Add data sampling parameters for this model
        data_params = {
            'sample_fraction': np.random.uniform(config.min_sample_fraction, config.max_sample_fraction),
            'feature_fraction': np.random.uniform(config.min_feature_fraction, config.max_feature_fraction),
            'bootstrap': np.random.choice([True, False]),
            'feature_importance_based': np.random.choice([True, False]),
        }
        
        model = xgb.XGBClassifier(**params)
        model_variations.append((model, params, data_params))
    
    return model_variations

# Sample data for a specific model variation
def sample_data_for_model(X_train, y_train, data_params, random_state=42):
    sample_size = int(len(X_train) * data_params['sample_fraction'])
    
    # Sample rows (data points)
    if data_params['bootstrap']:
        # Bootstrap sampling (with replacement)
        indices = resample(
            np.arange(len(X_train)), 
            replace=True, 
            n_samples=sample_size, 
            random_state=random_state,
            stratify=y_train  # Maintain class distribution
        )
    else:
        # Sampling without replacement
        indices = np.random.choice(
            np.arange(len(X_train)), 
            size=sample_size, 
            replace=False, 
            p=None  # Uniform probability
        )
    
    X_sampled = X_train.iloc[indices].copy()
    y_sampled = y_train.iloc[indices].copy()
    
    # Sample columns (features)
    n_features = X_sampled.shape[1]
    feature_size = int(n_features * data_params['feature_fraction'])
    
    if data_params['feature_importance_based'] and hasattr(X_train, 'model_feature_importance'):
        # Sample features based on feature importance
        feature_probs = X_train.model_feature_importance / X_train.model_feature_importance.sum()
        feature_indices = np.random.choice(
            np.arange(n_features), 
            size=feature_size, 
            replace=False, 
            p=feature_probs
        )
    else:
        # Random feature selection
        feature_indices = np.random.choice(
            np.arange(n_features), 
            size=feature_size, 
            replace=False
        )
    
    selected_features = X_sampled.columns[feature_indices]
    X_sampled = X_sampled[selected_features]
    
    return X_sampled, y_sampled, selected_features

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

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
    
    # Single train/validation split instead of cross-validation
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=config.validation_size, 
        random_state=config.seed, stratify=y
    )
    
    print(f'Train: {X_train_full.shape}')
    print(f'Validation: {X_val.shape}')
    
    # Create containers for predictions and metrics
    if test is not None:
        test_preds = np.zeros(shape=(X_test.shape[0], y.nunique()))
    
    # Get unique classes for proper model setup
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"Number of unique classes in data: {num_classes} {unique_classes}")
    
    # Ensure we're only doing binary classification
    if num_classes != 2:
        print("Warning: This model is designed for binary classification (model A wins vs model B wins)")
        print("The data should only contain wins (0) and losses (1), with no ties")
    
    # Generate model variations
    print(f"Generating {config.n_model_variations} XGBoost model variations...")
    model_variations = generate_model_variations(config.n_model_variations, config.seed)
    
    # Directory to save models
    os.makedirs("model_variations", exist_ok=True)
    
    # Dictionary to store feature sets used by each model
    feature_sets = {}
    
    # Dictionary to store model information for JSON export
    model_info = {}
    
    # Train all model variations
    model_cv_scores = []
    model_accuracy_scores = []
    
    for model_idx, (model, params, data_params) in enumerate(model_variations):
        model_key = f"var_{model_idx+1}"
        print(f"Training model variation {model_idx+1}/{config.n_model_variations}...")
        print(f"  Using {data_params['sample_fraction']*100:.1f}% of samples and {data_params['feature_fraction']*100:.1f}% of features")
        
        # Sample data for this model
        X_train, y_train, selected_features = sample_data_for_model(
            X_train_full, 
            y_train_full, 
            data_params, 
            random_state=config.seed + model_idx
        )
        
        print(f"  Sampled data shape: {X_train.shape}")
        
        # Store selected features for prediction later
        feature_sets[model_key] = selected_features
        
        # Train model
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val[selected_features], y_val)],
            verbose=False  # Reduced verbosity due to many models
        )
        
        # Evaluate model
        val_preds = model.predict_proba(X_val[selected_features])
        val_pred_labels = np.argmax(val_preds, axis=1)
        val_accuracy = np.mean(val_pred_labels == y_val)
        val_log_loss_score = log_loss(y_val, val_preds, labels=unique_classes)
        
        print(f"  Model {model_idx+1} Val log loss: {val_log_loss_score:.5f}, Val accuracy: {val_accuracy:.5f}")
        
        model_cv_scores.append(val_log_loss_score)
        model_accuracy_scores.append(val_accuracy)
        
        # Store model info for JSON
        model_info[model_key] = {
            "sample_percentage": data_params['sample_fraction'] * 100,
            "feature_percentage": data_params['feature_fraction'] * 100,
            "bootstrap": data_params['bootstrap'],
            "feature_importance_based": data_params['feature_importance_based'],
            "hyperparameters": params,
            "validation_log_loss": val_log_loss_score,
            "validation_accuracy": val_accuracy,
            "feature_count": len(selected_features),
            "sample_count": len(X_train)
        }
        
        # Save model
        model_path = f"model_variations/xgb_model_{model_key}.json"
        model.save_model(model_path)
        
        # Save feature set
        with open(f"model_variations/features_{model_key}.txt", 'w') as f:
            f.write('\n'.join(selected_features))
        
        # Generate predictions for test data
        if test is not None:
            # Use only features that were used during training
            test_features = [f for f in selected_features if f in X_test.columns]
            test_preds_model = model.predict_proba(X_test[test_features])
            # Add to ensemble with equal weight
            test_preds += test_preds_model / config.n_model_variations
        
        # Clean up to free memory
        gc.collect()
    
    # Print overall result
    print("="*90)
    print(f"Overall Log Loss: {np.mean(model_cv_scores):.5f}")
    print(f"Overall Accuracy: {np.mean(model_accuracy_scores):.5f}")
    
    # Convert NumPy types to Python types for JSON serialization
    model_info_serializable = convert_numpy_types(model_info)
    
    # Save model information to JSON file
    with open("model_variations/model_info.json", 'w') as f:
        json.dump(model_info_serializable, f, indent=4)
    print("Model information saved to model_variations/model_info.json")
    
    # Save predictions to submission file
    if test is not None and sample_submission is not None:
        for idx, t in enumerate(target_cols):
            sample_submission[t] = test_preds[:, idx]
        sample_submission.to_csv("submission.csv", index=False)
        print("Predictions saved to submission.csv")
    
    print("Done!")

def predict_preference(prompt, response_a, response_b):
    """
    Make predictions for a single example using all trained model variations.
    Returns an array of prefer_a_probability values.
    """
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
    from cleandata import Preprocessor
    preprocessor = Preprocessor()
    processed_data = preprocessor.run(test_data)
    
    # Prepare for prediction
    drop_cols = ["prompt", "response_a", "response_b", "winner_model_b"]
    X = processed_data.drop(columns=drop_cols, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Load all model variations and make predictions
    prefer_a_probabilities = []
    
    for model_idx in range(1, config.n_model_variations + 1):
        model_key = f"var_{model_idx}"
        model_path = f"model_variations/xgb_model_{model_key}.json"
        feature_path = f"model_variations/features_{model_key}.txt"
        
        if os.path.exists(model_path) and os.path.exists(feature_path):
            # Load model
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            # Load feature set
            with open(feature_path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            
            # Make sure all required features exist in the input data
            missing_features = [f for f in features if f not in X.columns]
            available_features = [f for f in features if f in X.columns]
            
            if missing_features:
                print(f"Warning: Model {model_key} is missing {len(missing_features)} features in the input data")
                if not available_features:
                    print(f"Skipping model {model_key} - no features available")
                    continue
            
            # Predict probabilities using only the features this model was trained on
            probs = model.predict_proba(X[available_features])[0]
            
            # Get probability of model A being preferred (converting from numpy float to Python float)
            prefer_a_probabilities.append(float(1 - probs[1]))
    
    return prefer_a_probabilities

if __name__ == "__main__":
    main()