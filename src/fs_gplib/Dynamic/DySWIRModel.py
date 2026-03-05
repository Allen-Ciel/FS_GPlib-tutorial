import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DySWIRModel(DiffusionModel):
    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 infection_kappa: float,
                 weakened_mu: float,
                 infection_nu: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         infection_kappa=infection_kappa,
                         weakened_mu=weakened_mu,
                         infection_nu=infection_nu,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)
        self.node_status['W_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)


    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()
        self.model = DySWIRModel_process(self.edge_index_list, self.edge_attr_list, self.infection_kappa, self.weakened_mu, self.infection_nu).to(self.device)

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
        x_list, R_mask_list, W_mask_list = self.model(self.node_status, iterations_times=times)

        out_all = torch.stack(x_list, dim=0), torch.stack(R_mask_list, dim=0), torch.stack(W_mask_list, dim=0)

        self.node_status['SI'], self.node_status['R_mask'], self.node_status['W_mask']= out_all[0][-1].squeeze(0), out_all[1][-1].squeeze(0), out_all[2][-1].squeeze(0)
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
                out_list, R_mask_list, W_mask_list = self.model(self.node_status, epoch_group)
                out_all = torch.stack(out_list, dim=0), torch.stack(R_mask_list, dim=0), torch.stack(W_mask_list, dim=0)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        out, R_mask, W_mask = out_all
        final = out.float()
        final[R_mask == True] = 3
        final[W_mask == True] = 2
        return final.squeeze(-1)

class DySWIRModel_process(Diffusion_process):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 infection_kappa,
                 weakened_mu,
                 infection_nu):
        super().__init__(edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         infection_kappa=infection_kappa,
                         weakened_mu=weakened_mu,
                         infection_nu=infection_nu)

    def forward(self, node_status, epochs=1, iterations_times=None):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        W_mask = node_status['W_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        x_list = []
        R_mask_list = []
        W_mask_list = []
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

            # S→I
            temp_i = self.propagate(edge_index, x=self.infection_kappa * (
                        x & ~W_mask).float()) * ~R_mask
            I_p = 1 - torch.exp(temp_i)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (i_rand_p < I_p) & (~x & ~R_mask)  # S

            # S→W
            temp_w = self.propagate(edge_index, x=self.weakened_mu * (x & ~W_mask).float()) * ~R_mask
            W_p = 1 - torch.exp(temp_w)
            w_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_w = (w_rand_p < W_p) & (~x & ~R_mask)

            # W→I
            temp_wi = self.propagate(edge_index, x=self.infection_nu * (
                        x & ~W_mask).float()) * ~R_mask
            WI_p = 1 - torch.exp(temp_wi)
            wi_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_wi = (wi_rand_p < WI_p) & W_mask

            # I→R
            mask_r = (x & ~W_mask)  # I

            x[mask_w] = True
            x[mask_i] = True
            x[mask_wi] = True
            x[mask_r] = False
            W_mask[mask_w] = True
            W_mask[mask_wi] = False
            R_mask[mask_r] = True

            self.times += 1
            x_list.append(x.clone())
            R_mask_list.append(R_mask.clone())
            W_mask_list.append(W_mask.clone())

        return x_list, R_mask_list, W_mask_list

    def message(self, x_j):
        return torch.log(1 - self.edge_attr * x_j)

