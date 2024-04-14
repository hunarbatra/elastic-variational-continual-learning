import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import random
import fire

from model_names_mapper import MODEL_NAMES_MAPPER


def map_model_names(model_data):
    model_data = {MODEL_NAMES_MAPPER[key]: value for key, value in model_data.items()}
    
    return model_data

def load_data(task_name, file_name, remove_models, experiments_dir):
    df = pd.read_csv(f'./{experiments_dir}/{task_name}/{file_name}.csv')
    if len(remove_models) > 0:
        df = df[~df['model'].isin(remove_models)]
    model_data = {}
    for model_name, group in df.groupby('model'):
        accuracies = group['avg_acc'].tolist()
        model_data[model_name] = accuracies
    
    sorted_models = sorted(model_data.keys(), key=lambda x: (
    not (x.startswith('evcl') or x.startswith('vcl_ewc')), 
    not x.startswith('vcl'), 
    not x.startswith('ewc'), 
    not x.startswith('coreset'), 
    len(x)
))
    sorted_model_data = {model: model_data[model] for model in sorted_models}
    
    model_data = map_model_names(sorted_model_data)
    
    return model_data

def plot_avg_accuracy(task_name, file_name, save_name=None, title='', ylim_low=None, remove_models=[], experiments_dir='./experiments'):
    model_data = load_data(task_name, file_name, remove_models, experiments_dir)
    
    plt.figure(figsize=(10, 4), dpi=100)
    plt.style.use(['no-latex', 'notebook'])

    for model_name, accuracies in model_data.items():
        line_style = '-'
        marker = 'o'
        if model_name.startswith('EWC'):
            line_style = '--'
            marker = '*'
        elif model_name.startswith('EVCL'):
            if 'Coreset' in model_name:
                line_style = '--'
                marker = 's'
            else:
                line_style = '-'
                marker = 's'
        elif model_name.startswith('VCL'):
            if 'Coreset' in model_name:
                line_style = '--'
                marker = 'o'
            else:
                line_style = '-'
                marker = 'o'
        elif 'Only' in model_name:
            line_style = '-.'
            marker = 'p'

        x = range(1, len(accuracies) + 1)
        plt.plot(x, accuracies, linestyle=line_style, marker=marker, label=model_name)

    plt.xlabel('# tasks', fontsize=10)
    plt.ylabel('Average Accuracy', fontsize=10)
    if len(title) > 0:
        plt.title(title, fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.xticks(range(1, len(x) + 1))  
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if ylim_low is not None:
        plt.ylim(ylim_low, 1.0)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'{experiments_dir}/{task_name}/{save_name}.pdf')
    plt.show()
        
    
if __name__ == '__main__':
    fire.Fire(plot_avg_accuracy)