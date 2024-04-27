# Abstract
Continual learning aims to train models on sequential tasks while balancing adaptation to new data and retention of existing knowledge. This work introduces Elastic Variational Continual Learning with Weight Consolidation (EVCL-WC), a novel hybrid model that integrates the variational posterior approximation mechanism of Variational Continual Learning (VCL) with the regularization-based parameter protection strategy of Elastic Weight Consolidation (EWC). By combining the strengths of both methods, EVCL-WC effectively mitigates catastrophic forgetting and enables better capture of dependencies between model parameters and task-specific data. Evaluated on five discriminative tasks, EVCL-WC consistently outperforms existing baselines in both domain-incremental and task-incremental learning scenarios for deep discriminative models.

# Steps to run:
## EVCL-WC:
```python
python3 run_evcl.py \
        --num_tasks=5 \
        --num_epochs=100 \ 
        --task_config='split_mnist' \ # supported task_configs: ['permuted_mnist', 'split_mnist', 'no_mnist', 'fashion_mnist', 'split_cifar']
        --experiment_name='experiment-name' \ # runs saved under f'experiments/{task_config}/{experiment_name}.csv'
        --ewc_lambda=100 # EWC regularization term λ
```

## VCL and VCL + Coresets:
```python
python3 run_vcl.py \
        --num_tasks=5 \
        --num_epochs=100 \
        --task_config='split_mnist' \ # supported task_configs: ['permuted_mnist', 'split_mnist', 'no_mnist', 'fashion_mnist', 'split_cifar']
        --coreset_size=200 \ # set to 0 if w/o coreset
        --coreset_method='random' \ # set to None if w/o coresets; supported coreset_methods: ['random', 'k-center', 'pca-k-center', 'class_balanced']
        --experiment_name='experiment-name' # runs saved under f'experiments/{task_config}/{experiment_name}.csv'
```
Note: set coreset_size = 0, and coreset_method = None, if running VCL without coreset.

## EWC:
```python
python3 run_ewc.py \
        --num_tasks=5 \
        --num_epochs=100 \
        --task_config='split_mnist' \ # supported task_configs: ['permuted_mnist', 'split_mnist', 'no_mnist', 'fashion_mnist', 'split_cifar']
        --experiment_name='experiment-name' \ # runs saved under f'experiments/{task_config}/{experiment_name}.csv'
        --ewc_lambda=100 \ # EWC regularization term λ
```

## Coreset Only:
```python
python3 coreset.py \
  --num_tasks=5 \
  --num_epochs=100 \
  --task_config='split_mnist' \
  --coreset_size=200 \
  --coreset_method='random' # supported coreset_methods: ['random', 'k-center', 'pca-k-center', 'class_balanced']
```

# Results
## PermutedMNIST:
![Permuted MNIST Plot](https://raw.githubusercontent.com/hunarbatra/elastic-variational-continual-learning/main/plots/permuted_mnist.png)
