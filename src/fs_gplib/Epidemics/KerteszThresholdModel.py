import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class KerteszThresholdModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 threshold: float,
                 adopter_rate: float,
                 percentage_blocked: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         threshold=threshold,
                         adopter_rate=adopter_rate,
                         percentage_blocked=percentage_blocked)


    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)

        if self.threshold == 0:
            self.node_status["node_threshold"] = torch.rand_like(self.node_status['SI'], dtype=torch.float32)
        else:
            self.node_status["node_threshold"] = torch.full_like(self.node_status['SI'].float(), self.threshold)

        self.node_status['B_mask'] = None

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = KerteszThreshold_process('sum', self.data.edge_index, self.threshold, self.adopter_rate, self.percentage_blocked, self.data.edge_attr)
        else:
            self.model = KerteszThreshold_process('mean', self.data.edge_index, self.threshold, self.adopter_rate,
                                                  self.percentage_blocked, None)

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
        self.node_status['SI'], self.node_status['B_mask'] = out_all[0].squeeze(0), out_all[1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self, iterations_times):
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times, batch_size=1):
        try:
            check_int(epochs=epochs)
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
                self.node_status['B_mask'] = None
                out_all = self.model(self.node_status, epoch_group)
                final = self._return_final(out_all)
                finals.append(final.to('cpu'))
            finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        out, B_mask = out_all
        final = out.float()
        final[B_mask] = -1
        return final.squeeze(-1)

class KerteszThreshold_process(Diffusion_process):
    def __init__(self,
                 aggr,
                 edge_index,
                 threshold,
                 adopter_rate,
                 percentage_blocked,

                 edge_attr=None):
        super().__init__(aggr=aggr,
                         edge_index=edge_index,
                         threshold=threshold,
                         adopter_rate=adopter_rate,
                         percentage_blocked=percentage_blocked,

                         edge_attr=edge_attr)

    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)

        if node_status['B_mask'] is None:
            S_index = torch.where(~node_status['SI'])[0]  # S状态节点的index
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

        while self.iterations_times > self.times:

            ai_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_ai = (~x & ~B_mask) & (ai_rand_p < self.adopter_rate)


            I_p = self.propagate(self.edge_index, x=x.float())*~B_mask

            mask_i = (~x) & (node_threshold <= I_p)

            x[mask_ai] = True
            x[mask_i] = True
            self.times += 1

        return x, B_mask

    def message(self, x_j):
        return self.edge_attr * x_j



