import torch
import pyro.distributions as dist

from tyxe.priors import Prior

from models.mlp import MLP
from utils.util import DEVICE   


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