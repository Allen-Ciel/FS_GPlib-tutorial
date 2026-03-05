import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class VoterModel(DiffusionModel):
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
        self.model = Voter_process(self.data.edge_index, 0)

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

class Voter_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 iterations_times):
        super(Voter_process, self).__init__(aggr='mean',
                         edge_index=edge_index,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]
        mask = torch.zeros_like(x, dtype=torch.bool)

        while self.iterations_times > self.times:
            # For each epoch, randomly select an index in the node dimension
            rand_indices = torch.randint(0, x.shape[1], (epochs,), device=x.device)

            # Obtain aggregated information
            temp = self.propagate(self.edge_index, x=x.float())

            # For each epoch, select the element in temp corresponding to the random index
            batch_indices = torch.arange(epochs, device=x.device)
            selected_temp = temp[batch_indices, rand_indices, :]  # shape: [epochs, 1]

            # Generate random values for each selected element
            random_vals = torch.rand_like(selected_temp)

            # Compare: if random value is less than the selected temp value, update to 1; else update to 0
            update_vals = (random_vals < selected_temp)

            # Update x at the positions indicated by batch_indices and rand_indices
            x[batch_indices, rand_indices, :] = update_vals

            self.times += 1
        return x
