import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DyThresholdModel(DiffusionModel):
    r"""Dynamic Threshold (DyThreshold) diffusion model on time-varying networks.

    This model extends the classical Threshold process to a sequence of graph
    snapshots :math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`. Each node is either
    inactive or active. At snapshot :math:`k`, an inactive node becomes active
    once the influence aggregated from its active neighbors in the **current**
    snapshot reaches its node threshold; active nodes remain active thereafter.

    The number of simulation steps cannot exceed ``len(edge_index_list)``.

    :param x: Node tensor of shape ``(N, 1)``.
    :type x: torch.Tensor
    :param edge_index_list: List of snapshot ``edge_index`` tensors, length
        :math:`T`.
    :type edge_index_list: list[torch.Tensor]
    :param seeds: Initially active nodes: list of node IDs or a float in
        ``[0,1)``.
    :type seeds: list[int] | float
    :param threshold: Node adoption threshold in ``[0,1]``. If ``threshold > 0``, 
        every node uses that same threshold value; if ``threshold == 0``, 
        node thresholds are sampled uniformly in :math:`[0,1)`;
        during batched multi-epoch execution, random thresholds are re-sampled
        independently for each epoch batch when ``threshold == 0``.

    :type threshold: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index. Defaults to
        ``'cpu'``.
    :type device: str | int
    :param rand_seed: *(optional)* Random seed used when *seeds* is a float.
        Defaults to ``None``.
    :type rand_seed: int | None
    :param edge_attr_list: *(optional)* Snapshot edge weights aligned with
        *edge_index_list*.
    :type edge_attr_list: list[torch.Tensor] | None
    """

    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 threshold: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         threshold=threshold,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)
        if self.threshold == 0:
            self.node_status["node_threshold"] = torch.rand_like(self.node_status['SI'], dtype=torch.float32)
        else:
            self.node_status["node_threshold"] = torch.full_like(self.node_status['SI'].float(), self.threshold)


    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()
        if self.edge_attr_list is None:
            self.model = DyThresholdModel_process('mean', self.edge_index_list, self.edge_attr_list, self.threshold).to(self.device)
        else:
            self.model = DyThresholdModel_process('sum', self.edge_index_list, self.edge_attr_list, self.threshold).to(self.device)

    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


    def run_iteration(self):
        """Advance the diffusion by one snapshot step.

        The internal ``node_status`` is updated so that subsequent calls
        continue from the latest state. Requires at least one remaining
        snapshot.

        :return: Node states after that step, shape ``(1, 1, N)``
            (values ``0`` or ``1``).
        :rtype: torch.Tensor
        """
        return self.run_iterations(1)

    def run_iterations(self, times):
        """Run *times* consecutive snapshot steps on the evolving graph sequence.

        The internal ``node_status`` is updated to the state after the last
        step. Requires ``len(edge_index_list) - t >= times`` where :math:`t` is
        the number of steps already consumed on this process.

        :param times: Number of snapshots to advance (must not exceed remaining
            snapshots).
        :type times: int
        :return: Node states after each step, stacked with shape
            ``(times, 1, N)`` (values ``0`` or ``1``).
        :rtype: torch.Tensor
        """
        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)
        # self.model._set_iterations(times)
        if len(self.edge_index_list) - self.model.times < times:
            raise ValueError('The number of remaining snapshots must be larger than iteration times')
        x_list = self.model(self.node_status, iterations_times=times)
        out_all = torch.stack(x_list, dim=0)
        self.node_status['SI'] = out_all[-1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self):
        """Run one Monte-Carlo realisation over the **full** snapshot sequence.

        The process internal step counter is reset; node states are
        **re-initialised** before the epoch starts.

        :return: Node states trajectory over all snapshots, shape ``(T, 1, N)``
            with :math:`T =` ``len(edge_index_list)`` (values ``0`` or ``1``).
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, 1)

    def run_epochs(self, epochs, batch_size=200):
        """Run multiple independent Monte-Carlo realisations in batches.

        For each realisation the snapshot index is reset to the beginning and
        the diffusion is evolved through **all** snapshots. Node states are
        **re-initialised** before the run.

        When ``threshold == 0``, random node thresholds are re-sampled
        independently for each epoch batch.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param batch_size: *(optional)* Parallel epochs per batch. Defaults to
            ``200``.
        :type batch_size: int
        :return: Node states trajectories for all realisations, shape
            ``(T, E, N)`` where :math:`T =` ``len(edge_index_list)`` and
            :math:`E` is *epochs* (values ``0`` or ``1``).
        :rtype: torch.Tensor
        """

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
                out_list = self.model(self.node_status, epoch_group)
                out_all = torch.stack(out_list, dim=0)
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        final = out_all.float()
        return final.squeeze(-1)

class DyThresholdModel_process(Diffusion_process):
    def __init__(self,
                 aggr,
                 edge_index_list,
                 edge_attr_list,
                 threshold):
        super().__init__(aggr=aggr,
                         edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         threshold=threshold)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        x_list = []
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

            I_p = self.propagate(edge_index, x=x.float())  # *D_in

            mask_i = (~x) & (node_threshold <= I_p)
            x[mask_i] = True
            self.times += 1
            x_list.append(x.clone())

        return x_list

    def message(self, x_j):
        return self.edge_attr * x_j
