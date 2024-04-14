import torch
import os
import pandas as pd

from sklearn.metrics import auc


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

RESULTS_SCHEMA = {'model': [], 'trained_on': [], 'tasks_acc': [], 'avg_acc': []}


def save_results(model_name, trained_on, prev_task_acc, avg_acc, data_name, experiment_name):
    dir_path = f"experiments/{data_name}"
    file_path = f"{dir_path}/{experiment_name}.csv"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.isfile(file_path):
        results_df = pd.DataFrame(columns=RESULTS_SCHEMA)
    else:
        results_df = pd.read_csv(file_path)

    new_row = pd.DataFrame({
        'model': [model_name],
        'trained_on': [trained_on],
        'tasks_acc': [prev_task_acc],
        'avg_acc': [avg_acc]
    })

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(file_path, index=False)
    
def get_model_name(base, coreset_size=0, coreset_method=None, model_suffix=None):
    model_name = f'{base}_{coreset_method}_{coreset_size}' if coreset_size > 0 else base
    
    if model_suffix is not None:
        return f'{model_name}_{model_suffix}'
    
    return model_name
    