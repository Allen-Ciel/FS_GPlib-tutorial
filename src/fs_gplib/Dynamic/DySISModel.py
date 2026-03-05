import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DySISModel(DiffusionModel):
    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 infection_beta: float,
                 recovery_lambda: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         infection_beta=infection_beta,
                         recovery_lambda=recovery_lambda,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)


    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()
        self.model = DySISModel_process(self.edge_index_list, self.edge_attr_list, self.infection_beta, self.recovery_lambda).to(self.device)

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
        if len(self.edge_index_list) - self.model.times < times:
            raise ValueError('The number of remaining snapshots must be larger than iteration times')
        x_list = self.model(self.node_status, iterations_times=times)
        out_all = torch.stack(x_list, dim=0)  # (时间片，batch_size，节点数量， 1)
        self.node_status = out_all[-1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self):
        return self.run_epochs(1, 1)

    def run_epochs(self, epochs, batch_size=200):


        try:
            check_int(epochs=epochs, batch_size=batch_size)
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
                self.model._set_iterations()
                out_list = self.model(self.node_status, epoch_group)
                out_all = torch.stack(out_list, dim=0)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        final = out_all.float()
        return final.squeeze(-1)

class DySISModel_process(Diffusion_process):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 infection_beta,
                 recovery_lambda):
        super().__init__(edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         infection_beta=infection_beta,
                         recovery_lambda=recovery_lambda)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status.unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        x_list = []
        if iterations_times is None:
            iterations_times = len(self.edge_index_list)
        else:
            iterations_times = iterations_times+int(self.times)
        self.device = x.device
        while iterations_times > self.times:
            edge_index = self.edge_index_list[self.times].to(self.device)
            if self.edge_attr_list is not None:
                self.edge_attr = self.edge_attr_list[self.times].to(self.device).unsqueeze(0).unsqueeze(2).to(self.device)  # .repeat(epochs, 1, 1)
            else:
                self.edge_attr = torch.tensor([1]).to(
                    self.device)  # torch.ones_like(self.edge_index[0]).unsqueeze(0).unsqueeze(2)

            temp = self.propagate(edge_index, x=x.float())
            I_p = 1 - torch.exp(temp)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (x == False) & (i_rand_p < I_p)  # S -> I

            s2_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_s2 = (x == True) & (s2_rand_p < self.recovery_lambda)  # I -> S

            x[mask_i] = True
            x[mask_s2] = False
            self.times += 1
            x_list.append(x.clone())

        return x_list

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

