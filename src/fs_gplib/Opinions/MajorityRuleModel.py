import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class MajorityRuleModel(DiffusionModel):
    r"""Binary Majority Rule opinion dynamics model on static graphs.

    Each node holds a binary opinion in ``{0, 1}``.  At every step, a group
    of :math:`q` nodes is sampled uniformly at random from the whole
    population with replacement.  The majority opinion within that sampled
    group is computed, and all sampled nodes adopt that majority opinion.  If
    the vote is tied, the tie is resolved in favour of opinion ``1``.

    Returned node states are encoded as: 0 = opinion 0, 1 = opinion 1.

    Unlike neighbor-based models such as Voter or Q-Voter, this
    implementation does not use local network neighborhoods in the update
    rule; the graph object is kept mainly for a consistent API and to provide
    the number of nodes.

    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``num_nodes`` (:math:`|V|`).
    :type data: torch_geometric.data.Data
    :param seeds: Initial nodes with opinion ``1``.  Pass a list of node IDs,
        a float in ``[0,1)`` to initialise that fraction of nodes chosen
        uniformly at random with opinion ``1``, or ``None``.
    :type seeds: list[int] | float | None
    :param q: Size of the randomly sampled discussion group.  Must be a
        positive integer.
    :type q: int
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
                 q,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         q=q,
                        )
    def _validate_parameters(self, kwargs):
        '''
        :param q:int >0
        '''
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
        self.model = MajorityRule_process(self.data.edge_index, self.q, 0)

    # def _set_seed(self, seeds):
    #     super()._initialize_seeds(seeds)
    #     self._init_node_status()


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

class MajorityRule_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 q,
                 iterations_times):
        super().__init__(
                         edge_index=edge_index,
                         q=q,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:

            sampled_indices = torch.randint(0, x.size(1), (epochs, self.q), device=x.device)

            sampled_values = x.squeeze(-1).gather(1, sampled_indices)

            true_counts = sampled_values.sum(dim=1)

            majority_true = (true_counts >= (self.q + 1) // 2).unsqueeze(-1)


            x_updated = x.clone().squeeze(-1)  # shape: [E, N]

            batch_indices = torch.arange(epochs, device=x.device).unsqueeze(1).expand(-1, self.q)  # shape: [E, q]
            x_updated[batch_indices, sampled_indices] = majority_true.expand(-1, self.q)
            x = x_updated.unsqueeze(-1)

            self.times += 1
        return x
