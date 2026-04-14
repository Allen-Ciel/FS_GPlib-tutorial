from tqdm import tqdm


from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SWIRModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 infection_kappa: float,
                 weakened_mu: float,
                 infection_nu: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_kappa=infection_kappa,
                         weakened_mu=weakened_mu,
                         infection_nu=infection_nu)

    def _init_node_status(self):
        self.node_status = dict()
        # mask
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        self.node_status['W_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)


    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = SWIR_process(self.data.edge_index, self.infection_kappa, self.weakened_mu, self.infection_nu, self.data.edge_attr)
        else:
            self.model = SWIR_process(self.data.edge_index, self.infection_kappa, self.weakened_mu, self.infection_nu,None)


    def run_iteration(self):
        return self.run_iterations(1)

    def run_iterations(self, times):
        check_int(times=times)

        self.model._set_iterations(times)
        out_all = self.model(self.node_status)
        self.node_status['SI'], self.node_status['W_mask'], self.node_status['R_mask'] = out_all[0].squeeze(0), out_all[1].squeeze(0), out_all[2].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self, iterations_times):
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times, batch_size=200):
        
        check_int(iterations_times=iterations_times, epochs=epochs, batch_size=batch_size)

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
        out, W_mask, R_mask = out_all
        final = out.float()
        final[R_mask == True] = 3
        final[W_mask == True] = 2
        return final.squeeze(-1)

# SWIR based on Message Passing
class SWIR_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_kappa,
                 weakened_mu,
                 infection_nu,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(edge_index=edge_index,
                         infection_kappa=infection_kappa,
                         weakened_mu=weakened_mu,
                         infection_nu=infection_nu,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)

    def forward(self, node_status, epochs=1):

        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1)

        W_mask = node_status['W_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)

        while self.times < self.iterations_times:
            # S→I
            temp_i = self.propagate(self.edge_index, x= self.infection_kappa * (x & ~W_mask).float()) * ~R_mask
            I_p = 1 - torch.exp(temp_i)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (i_rand_p < I_p) & (~x & ~R_mask)

            # S→W
            temp_w = self.propagate(self.edge_index, x=self.weakened_mu * (x & ~W_mask).float()) * ~R_mask
            W_p = 1 - torch.exp(temp_w)
            w_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_w = (w_rand_p < W_p) & (~x & ~R_mask)

            # W→I
            temp_wi = self.propagate(self.edge_index, x= self.infection_nu * (x & ~W_mask).float()) * ~R_mask
            WI_p = 1 - torch.exp(temp_wi)
            wi_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_wi = (wi_rand_p < WI_p) & W_mask

            # I→R
            mask_r = (x & ~W_mask)

            x[mask_w] = True
            x[mask_i] = True
            x[mask_wi] = True # 这个理论上不需要
            x[mask_r] = False
            W_mask[mask_w] = True
            W_mask[mask_wi] = False
            R_mask[mask_r] = True

            self.times += 1
        return x, W_mask, R_mask

    def message(self, x_j):

        return torch.log(1 - self.edge_attr * x_j)