import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class VoterModel(DiffusionModel):
    r"""Binary Voter opinion dynamics model on static graphs.

    Each node holds a binary opinion in ``{0, 1}``.  At every step, one node
    is selected uniformly at random and updates its opinion by copying a
    randomly chosen neighbor's opinion.  In this implementation, that update
    is realised by computing the fraction of neighbors with opinion ``1`` and
    then sampling a Bernoulli random variable with that probability.

    Returned node states are encoded as: 0 = opinion 0, 1 = opinion 1.

    Self-loops are removed internally so that a node does not influence its
    own opinion during the update.

    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``edge_index`` (the edge set :math:`E`)
        and ``num_nodes`` (:math:`|V|`).
    :type data: torch_geometric.data.Data
    :param seeds: Initial nodes with opinion ``1``.  Pass a list of node IDs,
        a float in ``[0,1)`` to initialise that fraction of nodes chosen
        uniformly at random with opinion ``1``, or ``None``.
    :type seeds: list[int] | float | None
    :param device: *(optional)* ``'cpu'`` or a CUDA device index.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param rand_seed: *(optional)* Random seed used when *seeds* is a
        float.  Defaults to ``None``.
    :type rand_seed: int | None
    """

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
        """Execute a single opinion-update step.

        The internal ``node_status`` is updated so that subsequent calls
        continue from the latest opinion configuration.

        :return: Node opinions after one step, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_iterations(1)


    def run_iterations(self, times):
        """Execute *times* opinion-update steps sequentially.

        The internal ``node_status`` is updated in-place so that subsequent
        calls continue from the latest opinion configuration.

        :param times: Number of steps to run.
        :type times: int
        :return: Node opinions at final step, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
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
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node opinions are **re-initialised** before the epoch starts.

        :param iterations_times: Number of opinion-update steps per epoch.
        :type iterations_times: int
        :return: Node opinions at final step of the epoch, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times, batch_size=200):
        """Run multiple independent Monte-Carlo epochs in batches.

        Node opinions are **re-initialised** before the run.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param iterations_times: Number of opinion-update steps per epoch.
        :type iterations_times: int
        :param batch_size: *(optional)* Number of epochs processed
            in parallel per batch.
            Defaults to ``200``.
        :type batch_size: int
        :return: Node opinions at final step of all epochs, shape ``(epochs, N)``.
        :rtype: torch.Tensor
        """

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
