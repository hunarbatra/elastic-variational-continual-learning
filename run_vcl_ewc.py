import torch
import pyro
import tyxe

import functools
import copy
import fire

import torch.nn.functional as F
import pyro.distributions as dist
import pyro.optim

from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from models.mle_prior import MLEPrior

from models.mlp import MLP
from trainer.trainer import train_mle
from data.data_generator import fetch_datasets
from utils.util import DEVICE, USE_CUDA, save_results, get_model_name
from utils.task_config import load_task_config
from coreset import update_coreset
from trainer.finetune import finetune_over_coreset

from typing import Optional, List
from tqdm import tqdm


def compute_fisher_info(bnn, data_loader, head_modules, n_samples=200, ewc_gamma=1.):
    est_fisher_info = {}
    for name, param in bnn.named_parameters():
        if not any(name.startswith(head) for head in head_modules):
            est_fisher_info[name] = param.detach().clone().zero_()
    
    bnn.net.eval()
    
    for index, (x, y) in enumerate(data_loader):
        if n_samples is not None and index > n_samples:
            break
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with torch.no_grad():
            output = bnn.predict(x, num_predictions=1, aggregate=False).squeeze(0)
        
        output.requires_grad = True
        label_weights = F.softmax(output, dim=1)
        
        for label_index in range(output.shape[1]):
            label = torch.full((x.size(0),), label_index, dtype=torch.long).to(DEVICE)
            negloglikelihood = F.cross_entropy(output, label) # NLL for the current class label
            bnn.zero_grad()
            # Compute the gradients of NLL wrt BNN parameters
            negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)
            # Accumulate the squared gradients weighted by the predicted class probabilities
            for name, param in bnn.named_parameters():
                if param.grad is not None and not any(name.startswith(head) for head in head_modules):
                    est_fisher_info[name] += (label_weights[:, label_index] * (param.grad.detach() ** 2)).sum()
    
    # Normalize the estimated Fisher information by the number of data points used for estimation
    est_fisher_info = {n: p / (index + 1) for n, p in est_fisher_info.items()}
    
    return est_fisher_info

class VariationalBNNWithEWC(tyxe.VariationalBNN):
    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None, ewc_lambda=0.0, fisher_info=None, prev_params=None):
        old_training_state = self.net.training
        self.net.train(True)
        
        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)
        
        def _as_tuple(x):
            if isinstance(x, (list, tuple)):
                return x
            return x,
        
        def _to(x, device):
            return map(lambda t: t.to(device) if device is not None else t, _as_tuple(x))
        
        for i in range(num_epochs):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                elbo += svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])
                
                if ewc_lambda > 0 and fisher_info is not None:
                    ewc_loss = 0
                    for name, param in self.named_parameters():
                        if name in fisher_info:
                            ewc_loss += (fisher_info[name] * (param - prev_params[name]) ** 2).sum()
                    num_data = len(data_loader.dataset)
                    elbo -= (ewc_lambda / num_data) * ewc_loss # minimise negative ELBO i.e maximise ELBO and minimise KL-div
                    # increasing ewc_lambda makes the network pay more attention to not deviating from previous tasks parameters
                    # we normalise ewc_lambda with the data size, to ensure that the impact of EWC regularisation term is proportional to the size of the current tasks dataset. Without this normalisation, the EWC term could dominate ELBO which is already normalised by the number of data points.
            
            if callback is not None and callback(self, i, elbo / num_batch):
                break
        
        self.net.train(old_training_state)
        return svi

def update_variational_approx(bnn, train_loader, curr_coreset, num_epochs, callback, ewc_lambda=0.0, fisher_info=None, prev_params=None, finetune_coreset=False):
    # update the variational approx
    if not finetune_coreset:
        non_coreset_data = list(set(train_loader.dataset) - set(curr_coreset))  
        data_loader = torch.utils.data.DataLoader(non_coreset_data, batch_size=train_loader.batch_size, shuffle=True)
    else:
        data_loader = torch.utils.data.DataLoader(curr_coreset, batch_size=train_loader.batch_size, shuffle=True)
    
    optim = pyro.optim.Adam({"lr": 1e-3})
    
    # apply local reparameterization trick for training VCL BNN
    with tyxe.poutine.local_reparameterization():
        bnn.fit(data_loader, optim, num_epochs, device=DEVICE, callback=callback, ewc_lambda=ewc_lambda, fisher_info=fisher_info, prev_params=prev_params)

def run_vcl_ewc(
    num_tasks: int = 5,
    num_epochs: int = 10,
    experiment_name: str = 'test',
    task_config: str = '',
    batch_size: int = 256,
    coreset_size: int = 0,
    coreset_method: str = 'random',
    finetune_method: Optional[str] = None,
    model_suffix: Optional[str] = '',
    ewc_lambda: float = 0.0,
    ewc_gamma: float = 1.0,
):
    input_dim, output_dim, hidden_sizes, single_head, data_name = load_task_config(task_config)
    train_loaders, test_loaders = fetch_datasets(batch_size, num_tasks, data_name)
    net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    net.to(DEVICE)
    num_heads = 1 if single_head else num_tasks
    
    # Train MLE network on task 0
    mle_net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
    mle_net.set_task(1)  # use the first task head for training/eval
    print(f"Current head being used for training MLE_NET - forward(): {mle_net.get_task()}")
    mle_acc = train_mle(mle_net, train_loaders[0], test_loaders[0], num_epochs)
    print(f'MLE Acc. after training on Task 1: {mle_acc}')
    
    # Initialize priors with MLE weights
    head_modules = [f"Head_{i+1}" for i in range(num_heads)]
    prior = MLEPrior(mle_net, head_modules, single_head)
    obs = tyxe.likelihoods.Categorical(-1)  # Bernoulli(-1, event_dim=1) for binary
    guide = functools.partial(
        tyxe.guides.AutoNormal,
        init_scale=1e-4,
        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(mle_net, prefix="net")  # init net with MLE priors
    )
    
    # Variational BNN
    bnn = VariationalBNNWithEWC(net, prior, obs, guide)  # convert net to BNN
    heads_list = [getattr(bnn.net, f"Head_{i+1}") for i in range(num_heads)]
    print(f"heads_list: {heads_list}")
    head_state_dicts = []
    for head in heads_list:
        head_state_dicts.append(copy.deepcopy(head.state_dict()))  # initialize head state for each head
    
    prev_coreset = []
    fisher_info = None
    prev_params = None
    
    for i, train_loader in enumerate(train_loaders, 1):
        # set the current head for training to the current task head
        head_idx = i if not single_head else 1
        bnn.net.set_task(head_idx)  # set current head for forward passes for training
        print(f"Current head being used for training bnn.net: {bnn.net.get_task()}")
        heads_list[head_idx-1].load_state_dict(head_state_dicts[head_idx-1])  # load head for current task (PyroLinear Head)
        
        # update coreset
        curr_coreset = update_coreset(prev_coreset, train_loader, coreset_size, coreset_method) if coreset_size else []
        
        elbos = []
        pbar = tqdm(total=num_epochs, unit="Epochs", postfix=f"Task {i}")
        
        def callback(_i, _ii, e):
            elbos.append(e / len(train_loader.sampler))  # Compute ELBO per data point
            pbar.update()
        
        obs.dataset_size = len(train_loader.sampler)
        
        # update the variational approximation for non-coreset data points (or for the curr task if curr_coreset = [])
        update_variational_approx(bnn, train_loader, curr_coreset, num_epochs, callback, ewc_lambda, fisher_info, prev_params)
        
        # Compute Fisher Information Matrix
        fisher_info = compute_fisher_info(bnn, train_loader, head_modules, ewc_gamma=ewc_gamma)
        prev_params = {name: param.detach().clone() for name, param in bnn.named_parameters() if not any(name.startswith(head) for head in head_modules)}
        
        head_state_dicts[head_idx-1] = copy.deepcopy(heads_list[head_idx-1].state_dict())  # save trained head
        
        if coreset_size > 0:
            if i == 1:
                # Initialize bnn_coreset with the same architecture as bnn
                bnn_coreset_net = MLP(input_dim, hidden_sizes, output_dim, num_tasks, single_head)
                bnn_coreset_net.to(DEVICE)
                bnn_coreset = VariationalBNNWithEWC(bnn_coreset_net, prior, obs, guide)
                bnn_coreset_heads_list = [getattr(bnn_coreset.net, f"Head_{i+1}") for i in range(num_heads)]
                bnn_coreset_head_state_dicts = [copy.deepcopy(head.state_dict()) for head in bnn_coreset_heads_list]
                bnn_coreset_fisher_info = None
                bnn_coreset_prev_params = None
            else:
                bnn_coreset_heads_list[head_idx-1].load_state_dict(bnn_coreset_head_state_dicts[head_idx-1])
            
            # Update the prior of bnn_coreset with the posterior from bnn's linear weights and biases -- recursive update
            site_names = [site for site in tyxe.util.pyro_sample_sites(bnn) if not any(site.startswith(head) for head in head_modules)]
            params_to_update = tyxe.priors.DictPrior({site: list(bnn.net_guide.get_detached_distributions(site).values())[0] for site in site_names})
            bnn_coreset.update_prior(params_to_update)
            
            # finetune the model on the coreset data
            bnn_coreset.net.set_task(head_idx)  # set the current task head for training bnn_coreset
            print(f"Current head being used for training bnn_coreset.net: {bnn_coreset.net.get_task()}")
            update_variational_approx(bnn_coreset, train_loader, curr_coreset, num_epochs, callback, ewc_lambda, bnn_coreset_fisher_info, bnn_coreset_prev_params, finetune_coreset=True)
            
            
            # Compute Fisher Information Matrix for bnn_coreset
            coreset_loader = torch.utils.data.DataLoader(curr_coreset, batch_size=train_loader.batch_size, shuffle=True)
            bnn_coreset_fisher_info = compute_fisher_info(bnn_coreset, coreset_loader, head_modules, ewc_gamma=ewc_gamma)
            bnn_coreset_prev_params = {name: param.detach().clone() for name, param in bnn_coreset.named_parameters() if not any(name.startswith(head) for head in head_modules)}
            
            bnn_coreset_head_state_dicts[head_idx-1] = copy.deepcopy(bnn_coreset_heads_list[head_idx-1].state_dict())  # update the bnn_coreset head for the current trained head for prediction
        
        pbar.close()
        
        print(f"Train over task {i} Accuracies:")
        prev_task_acc = []
        
        for j, test_loader in enumerate(test_loaders[:i], 1):
            # set the current head for eval (respective task head)
            eval_head_idx = j if not single_head else 1
            
            if coreset_size == 0:  # load bnn's eval head for testing
                bnn.net.set_task(eval_head_idx)  # set current tasks head for forward passes for evaluation
                print(f"Current head being used for evaluating bnn.net: {bnn.net.get_task()}")
                heads_list[eval_head_idx-1].load_state_dict(head_state_dicts[eval_head_idx-1])  # load head state for eval
            else:  # load bnn_coreset (finetuned model) eval head for testing
                bnn_coreset.net.set_task(eval_head_idx)
                print(f"Current head being used for evaluating bnn_coreset.net: {bnn_coreset.net.get_task()}")
                bnn_coreset_heads_list[eval_head_idx-1].load_state_dict(bnn_coreset_head_state_dicts[eval_head_idx-1])
            
            correct = 0
            total = 0
            
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                if coreset_size == 0:
                    preds = bnn.predict(x, num_predictions=8)
                else:
                    preds = bnn_coreset.predict(x, num_predictions=8)
                
                correct += (preds.argmax(-1) == y).sum().item()
                total += len(y)
            
            accuracy = correct / total
            print(f"Task {j} Accuracy: {accuracy:.4f}")
            prev_task_acc.append(accuracy)
        
        avg_acc = sum(prev_task_acc) / len(prev_task_acc)
        save_results(get_model_name('vcl_ewc', coreset_size, coreset_method, model_suffix), j, prev_task_acc, avg_acc, data_name, experiment_name, num_tasks)
        
        print(f"Train over task {i} avg: {avg_acc}")
        
        # propagate bnn posterior as the next prior (q_{t-1})
        site_names = [site for site in tyxe.util.pyro_sample_sites(bnn) if not any(site.startswith(head) for head in head_modules)]
        params_to_update = tyxe.priors.DictPrior({site: list(bnn.net_guide.get_detached_distributions(site).values())[0] for site in site_names})
        bnn.update_prior(params_to_update)
        
        # update the previous coreset
        prev_coreset = curr_coreset

if __name__ == '__main__':
    fire.Fire(run_vcl_ewc)