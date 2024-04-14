# ######## PERMUTED MNIST #########
# # VCL
# python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=0 --coreset_method=None --experiment_name='run4' &&
# # VCL + Random Coreset + 200
# python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run4' &&
# # VCL + K-Center Coreset + 200
# python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run4' &&
# # VCL + PCA-K-Center Coreset + 200
# python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='pca-k-center' --experiment_name='run4' &&
# # VCL+EWC
# python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=0 --coreset_method=None --experiment_name='run4' --ewc_lambda=100 &&
# # VCL+EWC + Random Coreset + 200
# python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run4' --ewc_lambda=100 &&
# # VCL+EWC + K-Center Coreset + 200
# python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run4' --ewc_lambda=100 &&
# # VCL+EWC + PCA-K-Center Coreset + 200
# python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='pca-k-center' --experiment_name='run4' --ewc_lambda=100 &&
# # Coreset only + Random Coreset + 200
# python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run4' &&
# # Coreset only + K-Center Coreset + 200
# python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run4' &&
# # Coreset only + PCA-K-Center Coreset + 200
# python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='pca-k-center' --experiment_name='run4'

##### NEW RUNS: 14/04 ######---
# VCL + Class Balanced Coreset + 200
python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='class_balanced' --experiment_name='run4' &&
# EVCL + Class Balanced Coreset + 200
python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='class_balanced' --experiment_name='run4' --ewc_lambda=100 &&
# Coreset only + Class Balanced Coreset + 200
python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='class_balanced' --experiment_name='run4' &&
# ECW Baseline
python3 run_ewc.py --num_tasks=10 --num_epochs=100 --task_config='permuted_mnist' --experiment_name="run4" --ewc_lambda=100 &&
# EVCL NEW
python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=0 --coreset_method=None --experiment_name='run4' --ewc_lambda=100 --model_suffix="new" &&
# VCL+EWC + Random Coreset + 200
python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run4' --ewc_lambda=100 --model_suffix="new" &&
# VCL+EWC + K-Center Coreset + 200
python3 run_evcl.py --num_tasks=5 --num_epochs=100 --task_config='permuted_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run4' --ewc_lambda=100 --model_suffix="new"