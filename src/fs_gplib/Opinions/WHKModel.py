import sys
from tqdm import tqdm
import torch_scatter
import random
import numpy as np


from .base import DiffusionModel, Diffusion_process
from ..utils import *


class WHKModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 epsilon,
                 weight,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         epsilon=epsilon,
                         weight=weight
                         )

    def _validate_parameters(self, kwargs):

        try:
            check_parameter(epsilon=kwargs['epsilon'])
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)


        if isinstance(kwargs['weight'], float):
            check_parameter(weight=kwargs['weight'])
        elif isinstance(kwargs['weight'], list):
            check_float_list(weight=kwargs['weight'])
        else:
            raise ValueError("Parameter weight must be a float or a list!")

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)
        self.weight = torch.tensor(kwargs['weight'])
    def _initialize_seeds(self, seeds):
        self.num_nodes = self._get_num_nodes(self.data)
        if seeds is None:

            random.seed(self.rand_seed)
            seeds = np.float32([random.uniform(-1, 1) for i in range(self.num_nodes)])
            self.seeds = seeds
        elif isinstance(seeds, list):
            if len(seeds) != self.num_nodes:
                raise ValueError('Number of seeds must equal the number of nodes')
            if max(seeds) >= 1 or min(seeds) <= -1:
                raise ValueError('Seeds must be between -1 and 1')
            self.seeds = seeds

    def _init_node_status(self):

        self.node_status = dict()
        self.node_status['SI'] = torch.tensor(self.seeds).unsqueeze(0).t().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self.weight = self.weight.to(self.device)

        self._init_node_status()
        self.model = WHK_process(self.data.edge_index, self.epsilon, self.weight, 0)


    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


    def run_iteration(self):
        return self.run_iterations(1)


    def run_iterations(self, times):
        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self.model._set_iterations(times)
        x = self.model(self.node_status)
        self.node_status['SI'] = x.squeeze(0)
        final = x.float()
        return final.squeeze(-1)

    def run_epoch(self, iterations_times):
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times, batch_size=200):
        try:
            check_int(iterations_times=iterations_times, epochs=epochs, batch_size=batch_size)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)
        self._init_node_status()
        epoch_groups = epochs_groups_list(epochs, batch_size)
        bar = tqdm(epoch_groups)
        finals = []
        with torch.no_grad():
            for i, epoch_group in enumerate(bar):
                bar.set_description('Batch {}'.format(i))
                self.model._set_iterations(iterations_times)
                out = self.model(self.node_status, epoch_group)
                final = out.float()
                final_cpu = final.squeeze(-1).to('cpu', non_blocking=True)  #
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=0)
        return finals

class WHK_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 epsilon,
                 weight,
                 iterations_times):
        super().__init__(aggr=None,
                         edge_index=edge_index,
                         epsilon=epsilon,
                         weight=weight,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
        self.mode = None
        if len(self.weight.shape):
            self.weight = self.weight.unsqueeze(0).unsqueeze(2)
        else:
            self.weight = self.weight.clone().detach().to(
                self.device)

    def forward(self, node_status, epochs=1):
        self.epochs = epochs
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:

            temp = self.propagate(self.edge_index, x=x).to(torch.float32)
            mask = ~torch.isnan(temp)
            pos_mask = torch.ones_like(temp)
            pos_mask[x>0] = -1
            temp += (1 + pos_mask*temp)*x
            x[mask.bool()] = temp[mask.bool()]
            self.times += 1
        return x

    def message(self, x_i, x_j):
        # x_i has shape [E, ] target
        # x_j has shape [E, ] source

        mask = abs(x_j - x_i) < self.epsilon

        return x_j * mask * self.weight, mask.float()

    def aggregate(self, inputs, ptr=None, dim_size=None):
        temp_count = inputs[1]
        temp_value = inputs[0]
        temp = torch_scatter.scatter_add(temp_value, self.edge_index[1], dim=1, dim_size=dim_size)/torch_scatter.scatter_add(temp_count, self.edge_index[1], dim=1, dim_size=dim_size)

        return temp
