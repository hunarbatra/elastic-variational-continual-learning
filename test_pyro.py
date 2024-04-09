import torch
import torch.utils.data as data

from torchvision import datasets, transforms

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(DEVICE)

def make_permuted_mnist_dataloaders(batch_size, num_tasks):
    train_loaders = []
    test_loaders = []
    for task in range(num_tasks):
        train_dataset = MNIST("./data", train=True, download=True, transform=tf.ToTensor())
        test_dataset = MNIST("./data", train=False, download=True, transform=tf.ToTensor())

        train_data = train_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
        train_targets = train_dataset.targets.to(DEVICE)
        test_data = test_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
        test_targets = test_dataset.targets.to(DEVICE)

        perm = torch.randperm(784, device=DEVICE)
        train_data = train_data[:, perm]
        test_data = test_data[:, perm]

        train_loaders.append(data.DataLoader(data.TensorDataset(train_data, train_targets),
                                             batch_size, shuffle=True, pin_memory=USE_CUDA))
        test_loaders.append(data.DataLoader(data.TensorDataset(test_data, test_targets),
                                            batch_size, shuffle=False, pin_memory=USE_CUDA))

    return train_loaders, test_loaders

def fetch_datasets(batch_size, num_tasks, data_name='permuted_mnist'):
    if data_name == 'permuted_mnist':
        return make_permuted_mnist_dataloaders(batch_size, num_tasks)
    
import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, input_dim, hidden_sizes, output_dim, num_tasks, single_head):
        super().__init__()
        self.input_size = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.single_head = single_head
        self.num_tasks = num_tasks if not self.single_head else 1

        prev_size = input_dim
        for idx, hidden_size in enumerate(hidden_sizes):
            self.add_module(f"Linear_{idx+1}", nn.Linear(prev_size, hidden_size))
            self.add_module(f"ReLU_{idx+1}", nn.ReLU(inplace=True))
            prev_size = hidden_size

        self.last_hidden_size = prev_size

        for task_id in range(self.num_tasks):
            self.add_module(f"Head_{task_id+1}", nn.Linear(prev_size, output_dim))
            
        self.current_task = 1
            
    def forward(self, x):
        for name, module in self.named_children():
            if name.startswith("Linear_") or name.startswith("ReLU_"):
                x = module(x)
        x = self.__getattr__(f"Head_{self.current_task}")(x)
        return x
    
    def set_task(self, task_id):
        self.current_task = task_id if not self.single_head else 1
        
    def get_task(self):
        return self.current_task

def train_mle(net, train_loader, test_loader, device, epochs):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy

import pyro.distributions as dist

from tyxe.priors import Prior


class MLEPrior(Prior):
    def __init__(self, mle_net, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mle_params = {}
        for name, param in mle_net.named_parameters():
            self.mle_params[name] = param.detach().to(DEVICE)

    def prior_dist(self, name, module, param):
        # fetch all modules weights and biases to be added as an MLE prior
        # while defining priors, we hide head modules to be copied over
        mle_param = self.mle_params[name]
        return dist.Normal(mle_param, torch.tensor(1.0, device=DEVICE))
    
import pandas as pd


RESULTS_SCHEMA = {'trained_on': [], 'prev_tasks_acc': [], 'avg_acc': []}

def save_results(trained_on, prev_task_acc, avg_acc, data_name, experiment_name):
    import os
    import pandas as pd

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
    
import argparse
import copy
import functools
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as tf
from torchvision.datasets import MNIST
import pyro
import pyro.distributions as dist
import tyxe


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")

class MLEPrior(Prior):
    def __init__(self, mle_net, head_modules, single_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mle_params = {}
        for name, param in mle_net.named_parameters():
            if single_head or not any(name.startswith(head.replace(".", "_")) for head in head_modules):
                self.mle_params[name] = param.detach().to(DEVICE)

        def expose_fn(module, name):
            return name in self.mle_params

        self.expose_fn = expose_fn

    def prior_dist(self, name, module, param):
        if name in self.mle_params:
            mle_param = self.mle_params[name]
            return dist.Normal(mle_param, torch.tensor(1.0, device=DEVICE))
        else:
            return super().prior_dist(name, module, param)

def run_vcl(input_dim, hidden_sizes, output_dim, num_tasks, num_epochs, single_head, experiment_name, data_name='permuted_mnist', batch_size=256):
    train_loaders, test_loaders = fetch_datasets(batch_size, num_tasks, data_name)

    net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    net.to(DEVICE)
    
    num_heads = 1 if single_head else num_tasks

    # Train MLE network on task 0
    mle_net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head) 
    mle_net.set_task(1) # use the first task head for training/eval
    print(f"Current head being used for training MLE_NET - forward(): {mle_net.get_task()}")
    mle_acc = train_mle(mle_net, train_loaders[0], test_loaders[0], DEVICE, num_epochs)
    print(f'MLE Acc. after training on Task 1: {mle_acc}')
    mle_heads = [getattr(mle_net, f"Head_{i+1}") for i in range(num_heads)]

    # Initialize priors with MLE weights
    # prior = MLEPrior(mle_net, expose_all=False) # fetch MLE prior head (has a single head)
    head_modules = [f"Head_{i+1}" for i in range(num_heads)]
    prior = MLEPrior(mle_net, head_modules, single_head)
    obs = tyxe.likelihoods.Categorical(-1) # Bernoulli(-1, event_dim=1) for binary
    guide = functools.partial(
        tyxe.guides.AutoNormal,
        init_scale=1e-4,
        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(mle_net, prefix="net") # init net with MLE priors
    )
    
    # variational bnn
    bnn = tyxe.VariationalBNN(net, prior, obs, guide) # convert net to BNN
    
    heads_list = [getattr(net, f"Head_{i+1}") for i in range(num_heads)]
    # move heads_list to device
    # heads_list.to(DEVICE)
    # print(f"heads_list: {heads_list}")
    
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict())) # initialize head state for each head
    
    for i, train_loader in enumerate(train_loaders, 1):
        head_idx = i if not single_head else 1 # set the current head for training to the current task head
        net.set_task(head_idx) # set current head for forward passes for training
        print(f"Current head being used for training NET - forward(): {net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1]) # load head for current task
        
        elbos = []
        pbar = tqdm(total=num_epochs, unit="Epochs", postfix=f"Task {i}")

        def callback(_i, _ii, e):
            elbos.append(e / len(train_loader.sampler))
            pbar.update()

        obs.dataset_size = len(train_loader.sampler)
        optim = pyro.optim.Adam({"lr": 1e-3})
        with tyxe.poutine.local_reparameterization():
            bnn.fit(train_loader, optim, num_epochs, device=DEVICE, callback=callback)

        pbar.close()
        head_state_dicts[head_idx-1] = copy.deepcopy(heads_list[head_idx-1].state_dict()) # save trained head

        print(f"Train over task {i} Accuracies:")
        prev_task_acc = []

        for j, test_loader in enumerate(test_loaders[:i], 1):
            eval_head_idx = j if not single_head else 1 # set the current head for eval (respective task head)
            net.set_task(eval_head_idx) # set current tasks head for forward passes for evaluation 
            print(f"Current head being used for evaluating NET - forward(): {net.get_task()}")
            heads_list[eval_head_idx-1].load_state_dict(head_state_dicts[eval_head_idx-1]) # load head state for eval
            
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
            
        avg_acc = sum(prev_task_acc)/len(prev_task_acc)
        save_results(j, prev_task_acc, avg_acc, data_name, experiment_name)
        print(f"Train over task {i} avg: {avg_acc}")
        
        if single_head:
            site_names = tyxe.util.pyro_sample_sites(bnn)
            bnn.update_prior(tyxe.priors.DictPrior(bnn.net_guide.get_detached_distributions(site_names)))
        else:
            site_names = [site for site in tyxe.util.pyro_sample_sites(bnn) if not any(site.startswith(head) for head in head_modules)]
            params_to_update = tyxe.priors.DictPrior({site: list(bnn.net_guide.get_detached_distributions(site).values())[0] for site in site_names})
            bnn.update_prior(params_to_update)
            

run_vcl(
input_dim=784, 
hidden_sizes=[100,100], 
output_dim=10,
num_tasks=5, 
num_epochs=20, 
single_head=True,
experiment_name='avg_runs_single_head_0804_100e_10t'
)