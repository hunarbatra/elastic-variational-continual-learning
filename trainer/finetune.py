import torch
import tyxe
import pyro

from utils.util import DEVICE    
        
        
def finetune_over_coreset(bnn_coreset, curr_coreset, num_epochs, callback=None, batch_size=256, finetune_method=None):
    # finetune the model over the coreset data point
    print('$$$$$$ finetuning bnn_coreset $$$$$$')
    coreset_loader = torch.utils.data.DataLoader(curr_coreset, batch_size=batch_size, shuffle=True)
    optim = pyro.optim.Adam({"lr": 1e-3})
    
    with tyxe.poutine.local_reparameterization():
        # for doing regular training without Variational Inference callback=None (such as for the Coreset only case)
        bnn_coreset.fit(coreset_loader, optim, num_epochs, device=DEVICE, callback=callback)
        
