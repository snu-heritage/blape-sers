# Utils to reproduce and evaluate the results of the BLaPE paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

# Label mapping dictionaries
BASE_LABELS = {
    'P': 'Hanji (Korean paper)',
    'C': 'Cotton', 
    'U': 'Unrefined silk',
    'S': 'Refined silk',
    'R': 'Ramie',
    'H': 'Hemp (linen)',
    'X': 'Pure dye sample'
}

DYE_LABELS = {
    '11': 'Sophora japonica (Goewhwa)',
    '12': 'Gardenia jasminoides (Chija)',
    '15': 'Phellodendron amurense (Hwangbyeok)',
    '21': 'Carthamus tinctorius (Honghwa)',
    '22': 'Caesalpinia sappan (Somok)',
    '23': 'Rubia akane (Kkokduseoni)',
    '24': 'Lithospermum erythrorhizon (Jacho)',
    '32': 'Fresh Persicaria tinctoria (Saengjjok)',
    '33': 'Fermented Persicaria tinctoria (Balhojjok)',
    '46': 'Quercus acutissima Cupule (Dotorigkakji)',
    '51': 'Rhus chinensis Gallnut (Obaeja)',
    '52': 'Punica granatum Peel (Seokryupi)'
}

MORDANT_LABELS = {
    'C': 'No Mordant',
    'Al': 'Aluminum',
    'Cu': 'Copper', 
    'Fe': 'Iron',
    'K': 'Potassium'
}

AGING_LABELS = {
    'B': 'Control (No aging)',
    'H288': 'Wet thermal',
    'D288': 'Dry thermal',
    'U288': 'Light (UV) irradiation'
}

LABEL_MAPPINGS = {
    'base': BASE_LABELS,
    'dye': DYE_LABELS,
    'mordant': MORDANT_LABELS,
    'aging': AGING_LABELS
}

def get_labels_from_code(code):
    """
    Extract labels from sample code.
    
    Args:
        code (str): Sample code in format {base}-{dye}-{mordant}-{aging}
        
    Returns:
        dict: Dictionary with keys 'base', 'dye', 'mordant', 'aging'
    """
    parts = code.split('-')
    if len(parts) != 4:
        raise ValueError(f"Invalid code format: {code}. Expected format: base-dye-mordant-aging")
    
    return {
        'base': parts[0],
        'dye': parts[1], 
        'mordant': parts[2],
        'aging': parts[3]
    }

def decode_labels(code):
    """
    Convert sample code to human-readable labels.
    
    Args:
        code (str): Sample code in format {base}-{dye}-{mordant}-{aging}
        
    Returns:
        dict: Dictionary with decoded labels
    """
    labels = get_labels_from_code(code)
    decoded = {}
    
    for category, value in labels.items():
        mapping = LABEL_MAPPINGS[category]
        decoded[category] = mapping.get(value, f"Unknown {category}: {value}")
    
    return decoded

def prepare_multilabel_data(data, feature_key='blape', regularization=None):
    """
    Prepare data for multilabel classification.
    
    Args:
        data (dict): Data dictionary with sample codes as keys
        feature_key (str): Key for features in each sample dict
        regularization (str): Regularization method ['l1', 'l2', 'minmax', 'stdnorm']
        
    Returns:
        tuple: (X, y_dict, label_encoders) where:
            - X: Feature matrix (n_samples, n_features)
            - y_dict: Dictionary of encoded labels for each category
            - label_encoders: Dictionary of LabelEncoder objects for each category
    """
    codes = list(data.keys())
    
    # Extract features
    X_list = []
    labels_list = []
    
    for code in codes:
        sample = data[code]
        if feature_key not in sample:
            continue
            
        features = sample[feature_key]
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        labels = get_labels_from_code(code)
        
        for i in range(features.shape[0]):
            X_list.append(features[i])
            labels_list.append(labels)
    
    X = np.array(X_list)
    
    # Apply regularization
    if regularization is not None:
        if regularization.lower() == 'l1':
            # L1 normalization (normalize each sample to unit L1 norm)
            X = X / (np.sum(np.abs(X), axis=1, keepdims=True) + 1e-8)
            X /= np.std(X)
        elif regularization.lower() == 'l2':
            # L2 normalization (normalize each sample to unit L2 norm)
            X = X / (np.sqrt(np.sum(X**2, axis=1, keepdims=True)) + 1e-8)
            X /= np.std(X)
        elif regularization.lower() == 'minmax':
            # MinMax scaling (scale features to [-1, 1])
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X = scaler.fit_transform(X)
        elif regularization.lower() == 'stdnorm':
            # Standard normalization (zero mean, unit variance)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            raise ValueError(f"Unknown regularization method: {regularization}. "
                           f"Available options: ['l1', 'l2', 'minmax', 'stdnorm']")
    
    # Prepare labels for each category
    categories = ['base', 'dye', 'mordant', 'aging']
    y_dict = {}
    label_encoders = {}
    
    for category in categories:
        category_labels = [labels[category] for labels in labels_list]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(category_labels)
        
        y_dict[category] = y_encoded
        label_encoders[category] = le
    
    return X, y_dict, label_encoders

def create_multilabel_pipeline(n_estimators=100, random_state=None):
    """
    Create Random Forest pipeline for multilabel classification.
    
    Args:
        n_estimators (int): Number of trees in the forest
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary of RandomForestClassifier for each category
    """
    categories = ['base', 'dye', 'mordant', 'aging']
    pipelines = {}
    
    for category in categories:
        pipelines[category] = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    
    return pipelines

def train_multilabel_models(X, y_dict, test_size=0.2, random_state=None):
    """
    Train multilabel classification models.
    
    Args:
        X (array): Feature matrix
        y_dict (dict): Dictionary of labels for each category
        test_size (float): Test set size ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (models, X_train, X_test, y_train_dict, y_test_dict)
    """
    first_category = list(y_dict.keys())[0]
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(stratified_split.split(X, y_dict[first_category]))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_dict = {}
    y_test_dict = {}
    
    for category, y in y_dict.items():
        y_train_dict[category] = y[train_idx]
        y_test_dict[category] = y[test_idx]
    
    models = create_multilabel_pipeline(random_state=random_state)
    
    for category, model in models.items():
        model.fit(X_train, y_train_dict[category])
    
    return models, X_train, X_test, y_train_dict, y_test_dict

def evaluate_multilabel_models(models, X_test, y_test_dict, label_encoders, verbose=True):
    """
    Evaluate multilabel classification models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (array): Test features
        y_test_dict (dict): Dictionary of test labels
        label_encoders (dict): Dictionary of label encoders
        verbose (bool): Whether to print detailed results
        
    Returns:
        dict: Dictionary of evaluation results for each category
    """
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    
    # Suppress UndefinedMetricWarning
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    
    results = {}
    
    for category, model in models.items():
        if verbose:
            print(f"\n=== {category.upper()} Classification Results ===")
        
        y_pred = model.predict(X_test)
        y_true = y_test_dict[category]
        
        # Get class names
        class_names = label_encoders[category].classes_
        
        # Classification report with zero_division handling
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     output_dict=True, zero_division=0)
        if verbose:
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results[category] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names,
            'accuracy': report['accuracy']
        }
    
    # Reset warning filters
    warnings.resetwarnings()
    
    return results