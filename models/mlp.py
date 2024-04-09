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