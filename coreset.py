import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import heapq

import torch
import pyro
import tyxe
import copy
import functools
from models.mlp import MLP
from data_generator import fetch_datasets
from utils import DEVICE, USE_CUDA, save_results
from task_config import load_task_config
from finetune import finetune_over_coreset
from tqdm import tqdm
from typing import Optional, List
import fire

import pyro.distributions as dist


def update_coreset(prev_coreset, train_loader, coreset_size, selection_method='random'):
    curr_task_data = list(train_loader.dataset)
    curr_task_data = random.sample(curr_task_data, min(coreset_size, len(curr_task_data)))
    combined_data = curr_task_data + prev_coreset if prev_coreset else curr_task_data
    
    if selection_method == 'random':
        curr_coreset = random.sample(combined_data, min(coreset_size, len(combined_data)))
    elif selection_method == 'k-center':
        curr_coreset = k_center_coreset(combined_data, coreset_size)
    elif selection_method == 'pca-k-center':
        curr_coreset = pca_k_center_coreset(combined_data, coreset_size)
    elif selection_method == 'vi':
        curr_coreset = vi_coreset(combined_data, min(coreset_size, len(combined_data)))
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")
    
    return curr_coreset

def k_center_coreset(data, coreset_size):
    data_array = np.array([x.numpy() for x, _ in data])
    num_points = len(data_array)

    # Initialize the coreset with a random data point
    coreset_indices = [random.randint(0, num_points - 1)]

    # Initialize a max-heap to store the distances and indices
    heap = [(-float('inf'), i) for i in range(num_points)]
    heapq.heapify(heap)

    # Iteratively select the farthest point from the current coreset
    for _ in range(coreset_size - 1):
        while True:
            _, farthest_point_index = heapq.heappop(heap)
            if farthest_point_index not in coreset_indices:
                break
        coreset_indices.append(farthest_point_index)

        # Update the distances in the max-heap
        for i in range(num_points):
            if i not in coreset_indices:
                dist = min(np.sum((data_array[farthest_point_index] - data_array[i])**2), -heap[0][0] if heap else float('inf'))
                if heap:
                    heapq.heapreplace(heap, (-dist, i))
                else:
                    heapq.heappush(heap, (-dist, i))

    return [data[i] for i in coreset_indices]

def pca_k_center_coreset(data, coreset_size):
    data_array = np.array([x.numpy() for x, _ in data])

    # Perform PCA to reduce the dimensionality
    pca = PCA(n_components=min(coreset_size, data_array.shape[1]))
    reduced_data = pca.fit_transform(data_array)
    num_points = len(reduced_data)

    # Initialize the coreset with a random data point
    coreset_indices = [random.randint(0, num_points - 1)]

    # Initialize a max-heap to store the distances and indices
    heap = [(-float('inf'), i) for i in range(num_points)]
    heapq.heapify(heap)

    # Iteratively select the farthest point from the current coreset
    for _ in range(coreset_size - 1):
        while True:
            _, farthest_point_index = heapq.heappop(heap)
            if farthest_point_index not in coreset_indices:
                break
        coreset_indices.append(farthest_point_index)

        # Update the distances in the max-heap
        for i in range(num_points):
            if i not in coreset_indices:
                dist = min(np.sum((reduced_data[farthest_point_index] - reduced_data[i])**2), -heap[0][0] if heap else float('inf'))
                if heap:
                    heapq.heapreplace(heap, (-dist, i))
                else:
                    heapq.heappush(heap, (-dist, i))

    return [data[i] for i in coreset_indices]

def run_coreset_only(
    num_tasks: int = 5,
    num_epochs: int = 10,
    experiment_name: str = 'test',
    task_config: str = '',
    batch_size: int = 256,
    coreset_size: int = 200,
    coreset_method: str = 'random'
):
    input_dim, output_dim, hidden_sizes, single_head, data_name = load_task_config(task_config)
    train_loaders, test_loaders = fetch_datasets(batch_size, num_tasks, data_name)
    net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    net.to(DEVICE)
    num_heads = 1 if single_head else num_tasks

    obs = tyxe.likelihoods.Categorical(-1)  # Bernoulli(-1, event_dim=1) for binary
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
    guide = None

    # Variational BNN
    bnn = tyxe.VariationalBNN(net, prior, obs, guide)  # convert net to BNN
    heads_list = [getattr(bnn.net, f"Head_{i+1}") for i in range(num_heads)]
    print(f"heads_list: {heads_list}")
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict()))  # initialize head state for each head

    prev_coreset = []
    for i, train_loader in enumerate(train_loaders, 1):
        # set the current head for training to the current task head
        head_idx = i if not single_head else 1
        bnn.net.set_task(head_idx)  # set current head for forward passes for training
        print(f"Current head being used for training bnn.net: {bnn.net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1])  # load head for current task (PyroLinear Head)

        # update coreset
        curr_coreset = update_coreset(prev_coreset, train_loader, coreset_size, coreset_method)
        
        # Callback function to compute the Evidence Lower Bound (ELBO) which is maximized during training
        # and to minimize the Kullback-Leibler (KL) divergence for VCL
        elbos = []
        pbar = tqdm(total=num_epochs, unit="Epochs", postfix=f"Task {i}")
        
        def callback(_i, _ii, e):
            elbos.append(e / len(train_loader.sampler))  # Compute ELBO per data point
            pbar.update()

        obs.dataset_size = len(train_loader.sampler)

        # finetune the model on the coreset data
        finetune_over_coreset(bnn, curr_coreset, num_epochs, callback=callback, batch_size=batch_size) 
        head_state_dicts[head_idx-1] = copy.deepcopy(heads_list[head_idx-1].state_dict())  # update the bnn head for the current trained head for prediction

        print(f"Train over task {i} Accuracies:")
        prev_task_acc = []
        for j, test_loader in enumerate(test_loaders[:i], 1):
            # set the current head for eval (respective task head)
            eval_head_idx = j if not single_head else 1
            bnn.net.set_task(eval_head_idx)  # set current tasks head for forward passes for evaluation
            print(f"Current head being used for evaluating bnn.net: {bnn.net.get_task()}")
            heads_list[eval_head_idx-1].load_state_dict(head_state_dicts[eval_head_idx-1])  # load head state for eval

            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = bnn.predict(x, num_predictions=8)
                correct += (preds.argmax(-1) == y).sum().item()
                total += len(y)

            accuracy = correct / total
            print(f"Task {j} Accuracy: {accuracy:.4f}")
            prev_task_acc.append(accuracy)

        avg_acc = sum(prev_task_acc) / len(prev_task_acc)
        save_results(j, prev_task_acc, avg_acc, data_name, experiment_name)
        print(f"Train over task {i} avg: {avg_acc}")

        # update the previous coreset
        prev_coreset = curr_coreset

if __name__ == '__main__':
    fire.Fire(run_coreset_only)