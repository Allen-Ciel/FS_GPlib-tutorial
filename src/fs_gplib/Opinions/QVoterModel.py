import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class QVoterModel(DiffusionModel):
    def __init__(self,
                 data,
                 seeds,
                 q,
                 epsilon,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         q=q,
                         epsilon=epsilon
                        )
    def _validate_parameters(self, kwargs):
        '''
        :param q:int >0
        :param epsilon:float [0,1)
        '''
        if kwargs['epsilon'] == 0:
            self.epsilon = 0
        else:
            try:
                check_parameter(epsilon=kwargs['epsilon'])
            except ValueError as e:
                print("Caught error:", e)
                sys.exit(1)
            self.epsilon = kwargs['epsilon']

        del kwargs['epsilon']

        try:
            check_int(**kwargs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)
        for param_name, value in kwargs.items():
            if value > 0:
                self.__setattr__(param_name, value)
            else:
                raise ValueError("Parameter q must be larger than 0!")

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        self.model = QVoter_process(self.data.edge_index, self.q, self.epsilon, 0)

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

class QVoter_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 q,
                 epsilon,
                 iterations_times):
        super().__init__(aggr='mean',
                         edge_index=edge_index,
                         q=q,
                         epsilon=epsilon,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:
            # For each epoch, randomly select an index in the node dimension
            rand_indices = torch.randint(0, x.shape[1], (epochs,), device=x.device)

            # Obtain aggregated information
            temp = self.propagate(self.edge_index, x=x.float())

            # For each epoch, select the element in temp corresponding to the random index
            batch_indices = torch.arange(epochs, device=x.device)
            selected_temp = temp[batch_indices, rand_indices, :]  # shape: [epochs, 1]

            # Generate random values for a tensor of shape [epochs, self.q]
            random_vals = torch.rand(epochs, self.q, device=x.device)

            # Broadcast selected_temp (shape [epochs, 1]) to compare with random_vals, yielding a candidate tensor of booleans of shape [epochs, self.q]
            candidate_vals = (random_vals < selected_temp)

            # For each epoch, check if all values in candidate_vals are identical (all True or all False)
            homogeneous = (torch.all(candidate_vals, dim=1) | torch.all(~candidate_vals, dim=1))

            # For epochs that are homogeneous, take the common value (first element) as the update value
            update_vals = candidate_vals[:, 0]  # shape: [epochs]

            # Only update x at positions where the homogeneous condition is met
            indices_to_update = homogeneous.nonzero(as_tuple=True)[0]
            if indices_to_update.numel() > 0:
                x[batch_indices[indices_to_update], rand_indices[indices_to_update], :] = update_vals[
                    indices_to_update].unsqueeze(-1)

            # For the remaining indices (non homogeneous), update with probability self.epsilon
            non_homogeneous_indices = (~homogeneous).nonzero(as_tuple=True)[0]
            if non_homogeneous_indices.numel() > 0:
                rand_vals_epsilon = torch.rand(non_homogeneous_indices.shape[0], device=x.device)
                update_rand_mask = rand_vals_epsilon < self.epsilon
                if update_rand_mask.any():
                    # Generate random boolean values (True/False) for the selected indices
                    random_bool = torch.randint(0, 2, (update_rand_mask.sum(),), device=x.device, dtype=torch.bool)
                    indices_to_random_update = non_homogeneous_indices[update_rand_mask]
                    x[batch_indices[indices_to_random_update], rand_indices[indices_to_random_update],
                    :] = random_bool.unsqueeze(-1)
            self.times += 1
        return x
