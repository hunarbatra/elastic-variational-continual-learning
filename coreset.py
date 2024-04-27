import torch
import pyro
import tyxe

import random
import copy
import functools
import heapq
import fire

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

import pyro.distributions as dist

from models.mlp import MLP
from data.data_generator import fetch_datasets
from utils.util import DEVICE, USE_CUDA, save_results, get_model_name
from utils.task_config import load_task_config
from trainer.finetune import finetune_over_coreset

from tqdm import tqdm
from typing import Optional, List


def update_coreset(prev_coreset, train_loader, coreset_size, selection_method='random', curr_idx=0):
    if isinstance(train_loader, list) and selection_method == 'class_balanced':
        tasks_so_far_data = []
        assert curr_idx > 0
        for i in range(0, curr_idx):
            tasks_so_far_data.append(train_loader[i])
        
        # Create a class-balanced list of combined_data for the total size of coreset_size
        combined_data = []
        num_tasks = len(tasks_so_far_data)
        samples_per_task = coreset_size // num_tasks
        for curr_task_loader in tasks_so_far_data:
            curr_task_data = list(curr_task_loader.dataset)
            combined_data.extend(random.sample(curr_task_data, samples_per_task))
        
        remaining_samples = coreset_size - len(combined_data)
        if remaining_samples > 0:
            combined_data.extend(random.sample(list(tasks_so_far_data[-1].dataset), remaining_samples))
        return combined_data
        
    elif isinstance(train_loader, torch.utils.data.dataloader.DataLoader):
        curr_task_data = list(train_loader.dataset)
        curr_task_data = random.sample(curr_task_data, min(coreset_size*2, len(curr_task_data))) # truncating current tasks data to {coreset_size * 2} to make it lil faster
        combined_data = curr_task_data + prev_coreset if prev_coreset else curr_task_data
    
    if selection_method == 'random':
        curr_coreset = random.sample(combined_data, min(coreset_size, len(combined_data)))
    elif selection_method == 'k-center':
        curr_coreset = k_center_coreset(combined_data, coreset_size)
    elif selection_method == 'pca-k-center':
        curr_coreset = pca_k_center_coreset(combined_data, coreset_size)
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")
    
    return curr_coreset

def k_center_coreset(data, coreset_size, via_pca=False):
    if not via_pca:
        data_array = np.array([x.cpu().numpy() for x, _ in data])
    else:
        data_array = data
        
    num_points = len(data_array)

    # Initialize the coreset with the first data point
    initial_index = 0  # deterministic start point
    coreset_indices = [initial_index]
    
    # Initialize the distances from the initial coreset point to all other points
    distances = np.full(num_points, np.inf)
    distances[initial_index] = 0
    for i in range(num_points):
        if i != initial_index:
            distances[i] = np.linalg.norm(data_array[i] - data_array[initial_index])
    
    # max-heap for maintaining max distances
    heap = [(-dist, i) for i, dist in enumerate(distances)]
    heapq.heapify(heap)

    # Iteratively select the farthest point from the current coreset
    while len(coreset_indices) < coreset_size:
        _, farthest_point_index = heapq.heappop(heap)
        if farthest_point_index not in coreset_indices:
            coreset_indices.append(farthest_point_index)
            
            # Update the distances and the heap for the remaining points
            for i in range(num_points):
                if i not in coreset_indices:
                    new_distance = np.linalg.norm(data_array[i] - data_array[farthest_point_index])
                    if new_distance < distances[i]:
                        distances[i] = new_distance
                        heap = [(-distances[j], j) for j in range(num_points) if j not in coreset_indices] # Rebuild the heap with updated distances
                        heapq.heapify(heap)

    if via_pca:
        return coreset_indices
    
    return [data[i] for i in coreset_indices]


def pca_k_center_coreset(data, coreset_size, n_components=20):
    data_array = np.array([x.cpu().numpy() for x, _ in data])
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_array)
    
    coreset_indices = k_center_coreset(reduced_data, coreset_size, via_pca=True)
    return [data[i] for i in coreset_indices]

def run_coreset_only(
    num_tasks: int = 5,
    num_epochs: int = 10,
    experiment_name: str = 'test',
    task_config: str = '',
    batch_size: int = 256,
    coreset_size: int = 200,
    coreset_method: str = 'random',
    model_suffix: Optional[str] = None,
    finetune_method: Optional[str] = None,
):
    input_dim, output_dim, hidden_sizes, single_head, data_name = load_task_config(task_config)
    train_loaders, test_loaders = fetch_datasets(batch_size, num_tasks, data_name)
    net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    net.to(DEVICE)
    num_heads = 1 if single_head else num_tasks

    obs = tyxe.likelihoods.Categorical(-1)  # Bernoulli(-1, event_dim=1) for binary
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
    guide = None

    mlp = tyxe.VariationalBNN(net, prior, obs, guide)  
    heads_list = [getattr(mlp.net, f"Head_{i+1}") for i in range(num_heads)]
    print(f"heads_list: {heads_list}")
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict()))  # initialize head state for each head

    prev_coreset = []
    for i, train_loader in enumerate(train_loaders, 1):
        # set the current head for training to the current task head
        head_idx = i if not single_head else 1
        mlp.net.set_task(head_idx)  # set current head for forward passes for training
        print(f"Current head being used for training mlp.net: {mlp.net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1])  # load head for current task (PyroLinear Head)

        # update coreset
        if coreset_size == 0:
            curr_coreset = [] 
        else: 
            if coreset_method == 'class_balanced':
                curr_coreset = update_coreset(prev_coreset, train_loaders, coreset_size, coreset_method, curr_idx=i)
            else: 
                curr_coreset = update_coreset(prev_coreset, train_loader, coreset_size, coreset_method) 

        obs.dataset_size = len(train_loader.sampler)

        # finetune the model on the coreset data
        finetune_over_coreset(mlp, curr_coreset, num_epochs, callback=None, batch_size=batch_size) 
        head_state_dicts[head_idx-1] = copy.deepcopy(heads_list[head_idx-1].state_dict())  # update the mlp head for the current trained head for prediction

        print(f"Train over task {i} Accuracies:")
        prev_task_acc = []
        for j, test_loader in enumerate(test_loaders[:i], 1):
            # set the current head for eval (respective task head)
            eval_head_idx = j if not single_head else 1
            mlp.net.set_task(eval_head_idx)  # set current tasks head for forward passes for evaluation
            print(f"Current head being used for evaluating mlp.net: {mlp.net.get_task()}")
            heads_list[eval_head_idx-1].load_state_dict(head_state_dicts[eval_head_idx-1])  # load head state for eval

            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = mlp.predict(x, num_predictions=8)
                correct += (preds.argmax(-1) == y).sum().item()
                total += len(y)

            accuracy = correct / total
            print(f"Task {j} Accuracy: {accuracy:.4f}")
            prev_task_acc.append(accuracy)

        avg_acc = sum(prev_task_acc) / len(prev_task_acc)
        save_results(get_model_name('coreset_only', coreset_size, coreset_method, model_suffix), j, prev_task_acc, avg_acc, data_name, experiment_name)
        print(f"Train over task {i} avg: {avg_acc}")

        # update the previous coreset
        prev_coreset = curr_coreset

if __name__ == '__main__':
    fire.Fire(run_coreset_only)