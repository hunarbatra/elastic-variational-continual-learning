###### SPLIT FASHION MNIST #######
# VCL
python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=0 --coreset_method=None --experiment_name='run1' &&
# VCL + Random Coreset + 200
python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run1' &&
# VCL + K-Center Coreset + 200
python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run1' &&
# VCL + PCA-K-Center Coreset + 200
python3 run_vcl.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run1' &&
# VCL+EWC
python3 run_vcl_ewc.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=0 --coreset_method=None --experiment_name='run1' --ewc_lambda=100 &&
# VCL+EWC + Random Coreset + 200
python3 run_vcl_ewc.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run1' --ewc_lambda=100 &&
# VCL+EWC + K-Center Coreset + 200
python3 run_vcl_ewc.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run1' --ewc_lambda=100 &&
# VCL+EWC + PCA-K-Center Coreset + 200
python3 run_vcl_ewc.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='pca-k-center' --experiment_name='run1' --ewc_lambda=100 &&
# Coreset only + Random Coreset + 200
python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='random' --experiment_name='run1' &&
# Coreset only + K-Center Coreset + 200
python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='k-center' --experiment_name='run1' &&
# Coreset only + PCA-K-Center Coreset + 200
python3 coreset.py --num_tasks=5 --num_epochs=100 --task_config='fashion_mnist' --coreset_size=200 --coreset_method='pca-k-center' --experiment_name='run1'
