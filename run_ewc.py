import torch
import pyro
import tyxe

import functools
import copy
import fire

import torch.nn.functional as F
import pyro.distributions as dist
import pyro.optim

from models.mlp import MLP
from data.data_generator import fetch_datasets
from utils.util import DEVICE, USE_CUDA, save_results, get_model_name
from utils.task_config import load_task_config

from typing import Optional, List
from tqdm import tqdm


def compute_fisher_info(mlp, prev_fisher_info, data_loader, head_modules, n_samples=5000, ewc_gamma=1.):
    est_fisher_info = {}
    for name, param in mlp.named_parameters():
        if not any(name.startswith(head) for head in head_modules):
            est_fisher_info[name] = param.detach().clone().zero_()
    
    mode = mlp.net.training
    mlp.net.eval()
    
    for index, (x, y) in enumerate(data_loader):
        if n_samples is not None and index > n_samples:
            break
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with torch.no_grad():
            output = mlp.predict(x, num_predictions=1, aggregate=False).squeeze(0)
        
        output.requires_grad = True
        label_weights = F.softmax(output, dim=1)
        
        for label_index in range(output.shape[1]):
            label = torch.full((x.size(0),), label_index, dtype=torch.long).to(DEVICE)
            negloglikelihood = F.cross_entropy(output, label) # NLL for the current class label
            mlp.zero_grad()
            # Compute the gradients of NLL wrt mlp parameters
            negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)
            # Accumulate the squared gradients weighted by the predicted class probabilities
            for name, param in mlp.named_parameters():
                if param.grad is not None and not any(name.startswith(head) for head in head_modules):
                    est_fisher_info[name] += (label_weights[:, label_index] * (param.grad.detach() ** 2)).sum()
    
    # Normalize the estimated Fisher information by the number of data points used for estimation
    est_fisher_info = {n: p / (index + 1) for n, p in est_fisher_info.items()}
    
    if prev_fisher_info is not None:
        for name, param in mlp.named_parameters():
            if name in prev_fisher_info:
                existing_values = prev_fisher_info[name]
                est_fisher_info[name] += ewc_gamma * existing_values
    
    mlp.net.train(mode)
    
    return est_fisher_info

class EWC(tyxe.VariationalBNN):
    def fit(self, data_loader, optim, num_epochs, num_particles=1, closed_form_kl=True, device=None, ewc_lambda=0.0, fisher_info=None, prev_params=None):
        old_training_state = self.net.training
        self.net.train(True)

        for i in range(num_epochs):
            total_loss = 0.
            num_batch = 1

            optim.zero_grad()  # Move the zero_grad() call outside the loop

            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                # Compute the loss using cross entropy
                output = self.net(input_data)
                loss = F.cross_entropy(output, observation_data)

                if ewc_lambda > 0 and fisher_info is not None:
                    ewc_loss = 0
                    for name, param in self.named_parameters():
                        if name in fisher_info:
                            ewc_loss += (fisher_info[name] * (param - prev_params[name]) ** 2).sum()
                    ewc_loss = (1./2) * ewc_loss
                    loss += ewc_lambda * ewc_loss

                loss.backward()
                total_loss += loss.item()

            optim.step()  # Move the optimizer step outside the loop

        self.net.train(old_training_state)
        return total_loss / num_batch

def train_ewc(mlp, train_loader, num_epochs, ewc_lambda, fisher_info=None, prev_params=None):
    # update the variational approx
    non_coreset_data = list(train_loader.dataset)  
    data_loader = torch.utils.data.DataLoader(non_coreset_data, batch_size=train_loader.batch_size, shuffle=True)
    
    optim = torch.optim.Adam(mlp.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    mlp.fit(data_loader, optim, num_epochs, device=DEVICE, ewc_lambda=ewc_lambda, fisher_info=fisher_info, prev_params=prev_params)

def run_ewc(
    num_tasks: int = 5,
    num_epochs: int = 10,
    experiment_name: str = 'test',
    task_config: str = '',
    batch_size: int = 256,
    model_suffix: Optional[str] = None,
    ewc_lambda: float = 100.0,
    ewc_gamma: float = 1.0,
):
    input_dim, output_dim, hidden_sizes, single_head, data_name = load_task_config(task_config)
    train_loaders, test_loaders = fetch_datasets(batch_size, num_tasks, data_name)
    net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    net.to(DEVICE)
    num_heads = 1 if single_head else num_tasks
    
    head_modules = [f"Head_{i+1}" for i in range(num_heads)]
    
    obs = tyxe.likelihoods.Categorical(-1)  # Bernoulli(-1, event_dim=1) for binary
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
    guide = None
    
    mlp = EWC(net, prior, obs, guide) 
    heads_list = [getattr(mlp.net, f"Head_{i+1}") for i in range(num_heads)]
    print(f"heads_list: {heads_list}")
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict()))  # initialize head state for each head
    
    prev_fisher_info = None
    prev_params = None
    
    for i, train_loader in enumerate(train_loaders, 1):
        # set the current head for training to the current task head
        head_idx = i if not single_head else 1
        mlp.net.set_task(head_idx)  # set current head for forward passes for training
        print(f"Current head being used for training mlp.net: {mlp.net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1])  # load head for current task (PyroLinear Head)
        
        obs.dataset_size = len(train_loader.sampler)
        
        train_ewc(mlp, train_loader, num_epochs, ewc_lambda, prev_fisher_info, prev_params)
        
        # Compute Fisher Information Matrix
        fisher_info = compute_fisher_info(mlp, prev_fisher_info, train_loader, head_modules, ewc_gamma=ewc_gamma)
        prev_params = {name: param.detach().clone() for name, param in mlp.named_parameters() if not any(name.startswith(head) for head in head_modules)}
        
        head_state_dicts[head_idx-1] = copy.deepcopy(heads_list[head_idx-1].state_dict())  # save trained head
        
        print(f"Train over task {i} Accuracies:")
        prev_task_acc = []
        
        for j, test_loader in enumerate(test_loaders[:i], 1):
            # set the current head for eval (respective task head)
            eval_head_idx = j if not single_head else 1
            
            # load mlp's eval head for testing
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
        save_results(get_model_name('evcl', model_suffix=model_suffix), j, prev_task_acc, avg_acc, data_name, experiment_name)
        
        print(f"Train over task {i} avg: {avg_acc}")
        
        # update the previous fisher info
        prev_fisher_info = fisher_info

if __name__ == '__main__':
    fire.Fire(run_ewc)
    
    
    # input_data, observation_data = tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0]
    #             input_tensor = input_data[0]  # Extract the input tensor from the tuple
    #             y_hat = self.net(input_tensor)
    #             loss = torch.nn.functional.cross_entropy(input=y_hat, target=observation_data, reduction='mean')

    #             if ewc_lambda > 0 and fisher_info is not None:
    #                 ewc_losses = []
    #                 for name, param in self.named_parameters():
    #                     if name in fisher_info:
    #                         fisher = fisher_info[name]
    #                         mean = prev_params[name]
    #                         ewc_losses.append((fisher * (param - mean)**2).sum())
    #                 ewc_loss = (1./2)*sum(ewc_losses)
    #                 total_loss += loss + ewc_lambda*ewc_loss
    #             else:
    #                 total_loss += loss