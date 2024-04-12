import torch
import tyxe
import pyro

from utils import DEVICE    
        
        
def finetune_over_coreset(bnn_coreset, curr_coreset, num_epochs, callback, batch_size):
    # finetune the model over the coreset data point
    print('$$$$$$ finetuning bnn_coreset $$$$$$')
    coreset_loader = torch.utils.data.DataLoader(curr_coreset, batch_size=batch_size, shuffle=True)
    optim = pyro.optim.Adam({"lr": 1e-3})
    
    with tyxe.poutine.local_reparameterization():
        bnn_coreset.fit(coreset_loader, optim, num_epochs, device=DEVICE, callback=callback)
