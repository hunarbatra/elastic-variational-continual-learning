import torch
import os

import pandas as pd


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

RESULTS_SCHEMA = {'trained_on': [], 'prev_tasks_acc': [], 'avg_acc': []}

def save_results(trained_on, prev_task_acc, avg_acc, data_name, experiment_name):
    dir_path = f"experiments/{data_name}"
    file_path = f"{dir_path}/{experiment_name}.csv"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.isfile(file_path):
        results_df = pd.DataFrame(columns=RESULTS_SCHEMA)
    else:
        results_df = pd.read_csv(file_path)

    new_row = pd.DataFrame({
        'trained_on': [trained_on],
        'prev_tasks_acc': [prev_task_acc],
        'avg_acc': [avg_acc]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(file_path, index=False)
    