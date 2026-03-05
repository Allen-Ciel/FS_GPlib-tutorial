import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DySEISModel(DiffusionModel):
    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 infection_beta: float,
                 removal_gamma: float,
                 latent_alpha: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)
        self.node_status['E_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()

        self.model = DySEISModel_process(self.edge_index_list, self.edge_attr_list, self.infection_beta, self.removal_gamma, self.latent_alpha).to(self.device)

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
        x_list, E_mask_list = self.model(self.node_status, iterations_times=times)

        out_all = torch.stack(x_list, dim=0), torch.stack(E_mask_list, dim=0)
        self.node_status['SI'], self.node_status['E_mask']= out_all[0][-1].squeeze(0), out_all[1][-1].squeeze(0)
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
                out_list, E_mask_list = self.model(self.node_status, epoch_group)


                out_all = torch.stack(out_list, dim=0), torch.stack(E_mask_list, dim=0)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        out, E_mask = out_all
        final = out.float()
        final[E_mask == True] = 2
        return final.squeeze(-1)


class DySEISModel_process(Diffusion_process):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 infection_beta,
                 removal_gamma,
                 latent_alpha):
        super().__init__(edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        E_mask = node_status['E_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        x_list = []
        E_mask_list = []
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

            temp = self.propagate(edge_index, x=(x & ~E_mask).float())
            E_p = 1 - torch.exp(temp)
            e_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_e = (e_rand_p < E_p) & (~x)  # S

            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = E_mask & (i_rand_p < self.latent_alpha)


            # I to S
            s2_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_s2 = (x & ~E_mask) & (s2_rand_p < self.removal_gamma)


            x[mask_e] = True
            x[mask_s2] = False
            E_mask[mask_e] = True
            E_mask[mask_i] = False
            self.times += 1
            x_list.append(x.clone())
            E_mask_list.append(E_mask.clone())

        return x_list, E_mask_list

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

