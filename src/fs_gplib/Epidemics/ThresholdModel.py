import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class ThresholdModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 threshold: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         threshold=threshold)


    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        if self.threshold == 0:
            self.node_status["node_threshold"] = torch.rand_like(self.node_status['SI'], dtype=torch.float32)
        else:
            self.node_status["node_threshold"] = torch.full_like(self.node_status['SI'].float(), self.threshold)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = Threshold_process("sum", self.data.edge_index, self.threshold, self.data.edge_attr)
        else:
            self.model = Threshold_process("mean",  self.data.edge_index, self.threshold, None)

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

    def run_epochs(self, epochs, iterations_times, batch_size=100):
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

class Threshold_process(Diffusion_process):
    def __init__(self,
                 aggr,
                 edge_index,
                 threshold,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(aggr=aggr,
                         edge_index=edge_index,
                         threshold=threshold,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)

    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)

        if epochs > 1 and self.threshold == 0:
            node_threshold = torch.rand_like(x, dtype=torch.float32)
        else:
            node_threshold = node_status["node_threshold"].expand(epochs, -1, -1)
        while self.iterations_times > self.times:

            I_p = self.propagate(self.edge_index, x=x.float())#*D_in

            mask_i = (~x) & (node_threshold <= I_p)
            x[mask_i] = True
            self.times += 1

            if not mask_i.any():
                break

        return x
    def message(self, x_j):

        return self.edge_attr * x_j




