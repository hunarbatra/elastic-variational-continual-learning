import random

def update_coreset(prev_coreset, train_loader, coreset_size):
    # combine curr_task_data and prev_coreset and sample coreset_size from them at random
    curr_task_data = list(train_loader.dataset)
    combined_data = curr_task_data + prev_coreset if prev_coreset else curr_task_data
    curr_coreset = random.sample(combined_data, min(coreset_size, len(combined_data)))
    
    return curr_coreset
