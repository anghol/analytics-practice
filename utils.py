import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def count_metrics(y_true, y_pred, multiclass=True, print_metrics=True):
    accuracy = accuracy_score(y_true, y_pred)
    if multiclass:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        precision = precision_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {'accuracy': accuracy, 'precision': precision, 'conf_matrix': conf_matrix}
    
    if print_metrics:
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('confusion_matrix:\n', conf_matrix)
    
    return metrics