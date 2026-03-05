import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SIModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 infection_beta: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_beta=infection_beta)


    def _init_node_status(self):
        self.node_status = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = SI_process(self.data.edge_index, self.infection_beta,  self.data.edge_attr)
        else:
            self.model = SI_process(self.data.edge_index, self.infection_beta, None)

    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


    def run_iteration(self):
        return self.run_iterations(1)


    def run_iterations(self, times: int):

        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self.model._set_iterations(times)
        out_all = self.model(self.node_status)
        self.node_status = out_all.squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self, iterations_times):
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs: int, iterations_times: int, batch_size: int=200):
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
                finals.append(final.to('cpu'))
            finals = torch.cat(finals, dim=0)
        return finals
    def _return_final(self, out_all):

        final = out_all.float()
        return final.squeeze(-1)

# SI based on Message Passing
class SI_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_beta,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(edge_index=edge_index,
                         infection_beta=infection_beta,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)


    def forward(self, node_status, epochs=1):
        x = node_status.unsqueeze(0).repeat(epochs, 1, 1)

        while self.iterations_times > self.times:
            temp = self.propagate(self.edge_index, x=x.float())
            I_p = 1 - torch.exp(temp)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (i_rand_p < I_p)
            x[mask_i] = True

            self.times += 1
        return x
    def message(self, x_j):

        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)
