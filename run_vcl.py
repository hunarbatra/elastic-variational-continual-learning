import pyro
import tyxe

import copy
import functools

from models.mlp import MLP
from models.mle_prior import MLEPrior

from trainer import train_mle
from data_generator import fetch_datasets
from utils import DEVICE, USE_CUDA, save_results
from task_config import load_task_config, check_task_config

from tqdm import tqdm
from typing import Optional, List

import fire


def run_vcl(
    num_tasks: int = 5, 
    num_epochs: int = 10, 
    experiment_name: str = 'test', 
    data_name: str = 'permuted_mnist', 
    task_config: Optional[str] = '', 
    batch_size: int = 256
):
    task_config = check_task_config(data_name, task_config)
    input_dim, output_dim, hidden_sizes, single_head = load_task_config(task_config)
    
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

    # Initialize priors with MLE weights
    head_modules = [f"Head_{i+1}" for i in range(num_heads)]
    prior = MLEPrior(mle_net, head_modules, single_head)
    obs = tyxe.likelihoods.Categorical(-1) # Bernoulli(-1, event_dim=1) for binary
    guide = functools.partial(
        tyxe.guides.AutoNormal,
        init_scale=1e-4,
        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(mle_net, prefix="net") # init net with MLE priors
    )
    
    # Variational BNN
    bnn = tyxe.VariationalBNN(net, prior, obs, guide) # convert net to BNN
    
    heads_list = [getattr(net, f"Head_{i+1}") for i in range(num_heads)]
    print(f"heads_list: {heads_list}")
    
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict())) # initialize head state for each head
    
    for i, train_loader in enumerate(train_loaders, 1):
        head_idx = i if not single_head else 1 # set the current head for training to the current task head
        net.set_task(head_idx) # set current head for forward passes for training
        print(f"Current head being used for training NET - forward(): {net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1]) # load head for current task (PyroLinear Head)
        
        elbos = []
        pbar = tqdm(total=num_epochs, unit="Epochs", postfix=f"Task {i}")

        # Callback function to compute the Evidence Lower Bound (ELBO) which is maximized during training
        # and to minimize the Kullback-Leibler (KL) divergence for VCL
        def callback(_i, _ii, e):
            elbos.append(e / len(train_loader.sampler))  # Compute ELBO per data point
            pbar.update()

        obs.dataset_size = len(train_loader.sampler)
        optim = pyro.optim.Adam({"lr": 1e-3})
        
        # apply local reparameterization trick for training VCL BNN
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
        
        site_names = [site for site in tyxe.util.pyro_sample_sites(bnn) if not any(site.startswith(head) for head in head_modules)]
        params_to_update = tyxe.priors.DictPrior({site: list(bnn.net_guide.get_detached_distributions(site).values())[0] for site in site_names})
        bnn.update_prior(params_to_update)

if __name__ == '__main__':
    fire.Fire(run_vcl)
