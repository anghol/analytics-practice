import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from typing import Union

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

def plot_loss_curve(loss_curve):
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.ylabel('loss')

def logging(name_model: str, metrics_default_model: dict, metrics_optim_model: Union[dict, None] = None, optim_params=None):
    with open('log.txt', 'a') as log_file:
        log_file.write(f'----- {name_model} -----\n')

        log_file.write('Metrics on default model\n')
        for (k, v) in metrics_default_model.items():
            log_file.write(f'{k}: {v}\n')
        log_file.write('\n')
        
        if metrics_optim_model:
            log_file.write('Metrics on optimal model\n')
            for (k, v) in metrics_optim_model.items():
                log_file.write(f'{k}: {v}\n')
            log_file.write('\n')
        
        if optim_params:
            log_file.write(f'Optimal parameters: {optim_params}\n\n')