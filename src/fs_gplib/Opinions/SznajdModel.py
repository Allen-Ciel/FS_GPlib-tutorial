import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SznajdModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                        )

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)

        self._init_node_status()
        self.model = Sznajd_process(self.data.edge_index, 0)

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

class Sznajd_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 iterations_times):
        super().__init__(aggr="max",
                         edge_index=edge_index,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:

            rand_indices = torch.randint(0, x.size(1), (epochs,), device=self.device)
            nei_mask = torch.zeros_like(x)
            nei_mask[torch.arange(epochs), rand_indices] = True
            nei = self.propagate(self.edge_index, x=nei_mask.float()).bool()
            nei += self.propagate(self.edge_index[[1, 0], :], x=nei_mask.float()).bool()


            rand_tensor = torch.rand_like(nei, dtype=torch.float)


            masked_rand = torch.where(nei, rand_tensor, torch.full_like(rand_tensor, -1.0))


            selected_indices = masked_rand.argmax(dim=1).squeeze(-1)

            node_u = rand_indices # [E]
            node_v = selected_indices  # [E]


            state_u = x[torch.arange(epochs), node_u, 0]  # [E]
            state_v = x[torch.arange(epochs), node_v, 0]  # [E]


            equal_mask = (state_u == state_v) # [E]，True 表示需要传播

            x_temp = x.clone().float()


            update_mask = torch.full((epochs, x.shape[1], 1), False, device=self.device)


            update_mask[torch.arange(epochs).to(self.device)[equal_mask], node_u[equal_mask], 0] = True
            update_mask[torch.arange(epochs).to(self.device)[equal_mask], node_v[equal_mask], 0] = True


            x_temp[~update_mask] = -1.0

            temp = self.propagate(self.edge_index, x=x_temp)

            mask = (temp != -1)
            x[mask] = temp[mask].bool()

            self.times += 1
        return x
