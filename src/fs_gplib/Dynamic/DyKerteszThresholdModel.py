import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DyKerteszThresholdModel(DiffusionModel):
    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 threshold: float,
                 adopter_rate: float,
                 percentage_blocked: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         threshold=threshold,
                         adopter_rate=adopter_rate,
                         percentage_blocked=percentage_blocked,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)
        if self.threshold == 0:
            self.node_status["node_threshold"] = torch.rand_like(self.node_status['SI'], dtype=torch.float32)
        else:
            self.node_status["node_threshold"] = torch.full_like(self.node_status['SI'].float(), self.threshold)
        self.node_status['B_mask'] = None

    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()
        if self.edge_attr_list is None:
            self.model = DyKerteszThresholdModel_process('mean', self.edge_index_list, self.edge_attr_list, self.threshold, self.adopter_rate, self.percentage_blocked).to(self.device)
        else:
            self.model = DyKerteszThresholdModel_process('sum', self.edge_index_list, self.edge_attr_list, self.threshold, self.adopter_rate, self.percentage_blocked).to(self.device)

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
        x_list, B_mask_list = self.model(self.node_status, iterations_times=times)

        out_all = torch.stack(x_list, dim=0), torch.stack(B_mask_list, dim=0)
        self.node_status['SI'], self.node_status['B_mask'] = out_all[0][-1].squeeze(0), out_all[1][-1].squeeze(0)
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
                out_list, B_mask_list = self.model(self.node_status, epoch_group)
                out_all = torch.stack(out_list, dim=0), torch.stack(B_mask_list, dim=0)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        out, B_mask = out_all
        final = out.float()
        final[B_mask] = -1
        return final.squeeze(-1)


class DyKerteszThresholdModel_process(Diffusion_process):
    def __init__(self,
                 aggr,
                 edge_index_list,
                 edge_attr_list,
                 threshold,
                 adopter_rate,
                 percentage_blocked
                 ):
        super().__init__(aggr=aggr,
                         edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         threshold=threshold,
                         adopter_rate=adopter_rate,
                         percentage_blocked=percentage_blocked)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        x_list = []
        B_mask_list = []
        if node_status['B_mask'] is None:
            S_index = torch.where(~node_status['SI'])[0]
            n_block = int(node_status['SI'].numel() * self.percentage_blocked)
            n_block = min(n_block, S_index.size(0))
            B_mask = torch.stack([

                torch.zeros(node_status['SI'].numel(), dtype=torch.bool, device=x.device)
                .index_fill(0, S_index[torch.randperm(S_index.size(0))[:n_block]], True)
                .view(node_status['SI'].shape)
                for _ in range(epochs)
            ])
        else:
            B_mask = node_status['B_mask'].expand(epochs, -1, -1)

        if epochs > 1 and self.threshold == 0:
            node_threshold = torch.rand_like(x, dtype=torch.float32)
        else:
            node_threshold = node_status["node_threshold"].expand(epochs, -1, -1)

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


            ai_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_ai = (~x & ~B_mask) & (ai_rand_p < self.adopter_rate)


            I_p = self.propagate(edge_index, x=x.float()) * ~B_mask

            mask_i = (~x) & (node_threshold <= I_p)

            x[mask_ai] = True
            x[mask_i] = True
            self.times += 1
            x_list.append(x.clone())
            B_mask_list.append(B_mask.clone())

        return x_list, B_mask_list

    def message(self, x_j):
        return self.edge_attr * x_j

