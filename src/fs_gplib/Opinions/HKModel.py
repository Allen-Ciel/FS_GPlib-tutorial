import sys
from tqdm import tqdm
import torch_scatter
import random

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class HKModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 epsilon,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         epsilon=epsilon
                        )

    def _initialize_seeds(self, seeds):
        self.num_nodes = self._get_num_nodes(self.data)
        if seeds is None:
            random.seed(self.rand_seed)
            seeds = [random.uniform(-1, 1) for i in range(self.num_nodes)]
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
        # pdb.set_trace()
        self._init_node_status()
        self.model = HK_process(self.data.edge_index, self.epsilon,0)

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
        out_all = self.model(self.node_status)
        self.node_status['SI'] = out_all.squeeze(0)
        final = self._return_final(out_all)
        return final

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
                out_all = self.model(self.node_status, epoch_group)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu', non_blocking=True)#
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        final = out_all.float()
        return final.squeeze(-1)

class HK_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 epsilon,
                 iterations_times):
        super().__init__(aggr=None,
                         edge_index=edge_index,
                         epsilon=epsilon,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
        self.mode = None
    def forward(self, node_status, epochs=1):
        self.epochs=epochs
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:
            temp = self.propagate(self.edge_index, x=x)
            mask = ~torch.isnan(temp)
            x[mask.bool()] = temp[mask.bool()]
            self.times += 1

        return x

    def message(self, x_i, x_j):

        mask = abs(x_j-x_i)<self.epsilon

        return x_j*mask, mask.float()

    def aggregate(self, inputs, ptr=None, dim_size=None):
        temp_count = inputs[1]
        temp_value = inputs[0]
        temp = torch_scatter.scatter_add(temp_value, self.edge_index[1], dim=1, dim_size=dim_size)/torch_scatter.scatter_add(temp_count, self.edge_index[1], dim=1, dim_size=dim_size)

        return temp

