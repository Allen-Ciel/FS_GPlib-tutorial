from tqdm import tqdm
import torch_scatter
import random
from torch_geometric.utils import remove_self_loops

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class HKModel(DiffusionModel):
    r"""Hegselmann-Krause (HK) bounded-confidence opinion dynamics on static graphs.

    Each node carries a continuous opinion :math:`h \in (-1, 1)` (initialised
    from *seeds* or drawn uniformly at random when *seeds* is ``None``).  At
    each step, for every node :math:`i` the set of *confidence neighbors*
    consists of graph neighbors :math:`j` with
    :math:`|h_i^{(k-1)} - h_j^{(k-1)}| < \varepsilon`.  If that set is
    non-empty, :math:`h_i^{(k)}` becomes the average of their opinions; if it
    is empty, :math:`h_i` stays unchanged.

    Self-loops are removed from the edge index.

    :param data: PyTorch Geometric ``Data`` representing :math:`G=(V,E)`.
        Must provide ``edge_index`` and ``num_nodes``.
    :type data: torch_geometric.data.Data
    :param seeds: Initial opinion per node, length ``num_nodes``, each strictly
        between ``-1`` and ``1``; or ``None`` to sample each component
        independently from :math:`\mathrm{Uniform}(-1, 1)` (using *rand_seed*
        for the RNG).
    :type seeds: list[float] | None
    :param epsilon: Confidence bound :math:`\varepsilon` in ``[0, 1]``; only
        neighbors with opinion separation below this threshold contribute to
        the update.
    :type epsilon: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param rand_seed: *(optional)* Seed for the random number generator when
        *seeds* is ``None``.  Defaults to ``None``.
    :type rand_seed: int | None
    """

    def __init__(self,
                 data,
                 seeds,
                 epsilon,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         epsilon=epsilon
                        )

    def _initialize_seeds(self, seeds):
        self.num_nodes = self._get_num_nodes(self.data)
        if seeds is None:
            random.seed(self.rand_seed)
            seeds = [random.uniform(-1, 1) for i in range(self.num_nodes)]
            self.seeds = seeds
        elif isinstance(seeds, list):
            if len(seeds) != self.num_nodes:
                raise ValueError('Number of seeds must equal the number of nodes')
            if max(seeds) >= 1 or min(seeds) <= -1:
                raise ValueError('Seeds must be between -1 and 1')
            self.seeds = seeds
        else:
            raise ValueError('Seeds must be a list of floats or None')

    def _init_node_status(self):

        self.node_status = dict()
        self.node_status['SI'] = torch.tensor(self.seeds).unsqueeze(0).t().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        # pdb.set_trace()
        self._init_node_status()
        self.model = HK_process(self.data.edge_index, self.epsilon,0)

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
        check_int(times=times)

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
                final_cpu = final.to('cpu', non_blocking=True)#
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        final = out_all.float()
        return final.squeeze(-1)

class HK_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 epsilon,
                 iterations_times):
        super().__init__(aggr=None,
                         edge_index=edge_index,
                         epsilon=epsilon,
                         iterations_times=iterations_times)

        self.edge_index, _ = remove_self_loops(edge_index)
        self.mode = None
    def forward(self, node_status, epochs=1):
        self.epochs=epochs
        x = node_status['SI'].repeat(epochs, 1, 1)  # [E, N, 1]

        while self.iterations_times > self.times:
            temp = self.propagate(self.edge_index, x=x)
            mask = ~torch.isnan(temp)
            x[mask.bool()] = temp[mask.bool()]
            self.times += 1

        return x

    def message(self, x_i, x_j):

        mask = abs(x_j-x_i)<self.epsilon

        return x_j*mask, mask.float()

    def aggregate(self, inputs, ptr=None, dim_size=None):
        temp_count = inputs[1]
        temp_value = inputs[0]
        temp = torch_scatter.scatter_add(temp_value, self.edge_index[1], dim=1, dim_size=dim_size)/torch_scatter.scatter_add(temp_count, self.edge_index[1], dim=1, dim_size=dim_size)

        return temp

