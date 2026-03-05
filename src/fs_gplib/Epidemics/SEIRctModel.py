import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SEIRctModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 infection_beta: float,
                 removal_gamma: float,
                 latent_alpha: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha)


    def _init_node_status(self):
        self.node_status = dict()
        # mask
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        self.node_status['E_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['E_iteration']= torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)
        self.node_status['I_iteration'] = torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = SEIRct_process(self.data.edge_index, self.infection_beta, self.removal_gamma, self.latent_alpha, self.data.edge_attr)
        else:
            self.model = SEIRct_process(self.data.edge_index, self.infection_beta, self.removal_gamma, self.latent_alpha, None)

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
        # x, E_mask, R_mask, E_iteration, I_iteration = self.model(self.node_status)
        out_all = self.model(self.node_status)
        self.node_status['SI'], self.node_status['E_mask'], self.node_status['R_mask'], self.node_status['E_iteration'], self.node_status['I_iteration'] = out_all[0].squeeze(0), out_all[1].squeeze(0), out_all[2].squeeze(0), out_all[3].squeeze(0), out_all[4].squeeze(0)
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
                finals.append(final.to('cpu'))
        finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        out, R_mask, E_mask, _, _ = out_all
        final = out.float()
        final[R_mask == True] = 3
        final[E_mask == True] = 2
        return final.squeeze(-1)

# SEIRct based on Message Passing
class SEIRct_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_beta,
                 removal_gamma,
                 latent_alpha,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(edge_index=edge_index,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)

    def forward(self, node_status, epochs=1):

        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1)

        E_mask = node_status['E_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)

        E_iteration = node_status['E_iteration'].unsqueeze(0).repeat(epochs, 1, 1)
        I_iteration = node_status['I_iteration'].unsqueeze(0).repeat(epochs, 1, 1)

        while self.times < self.iterations_times:

            temp = self.propagate(self.edge_index, x=(x & ~E_mask).float()) * ~R_mask
            E_p = 1 - torch.exp(temp)
            e_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_e = (e_rand_p < E_p) & (~x & ~R_mask)
            E_iteration[mask_e] = self.times


            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            I_p = 1 - torch.exp(-(self.times - E_iteration) * self.latent_alpha)
            mask_i = E_mask & (i_rand_p < I_p)
            I_iteration[mask_i] = self.times

            r_rand_p = torch.rand_like(x, dtype=torch.float32)
            R_p = 1 - torch.exp(-(self.times - I_iteration) * self.removal_gamma)
            mask_r = (x & ~E_mask) & (r_rand_p < R_p)

            x[mask_r] = False
            x[mask_e] = True
            E_mask[mask_e] = True
            E_mask[mask_i] = False
            R_mask[mask_r] = True

            self.times += 1

        return x, E_mask, R_mask, E_iteration, I_iteration

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

