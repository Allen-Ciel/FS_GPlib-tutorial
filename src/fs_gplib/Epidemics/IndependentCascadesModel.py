import sys
from tqdm import tqdm
from torch_geometric.utils import degree

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class IndependentCascadesModel(DiffusionModel):
    r"""Independent Cascades (IC) diffusion model on static graphs.

    Each node starts as inactive (susceptible) or active (seed).  At every
    step each newly active node gets exactly one chance to activate each of
    its inactive out-neighbors independently.  After that attempt the node
    can not activate any other nodes, but it remains *active* in the
    final cascade outcome.

    Returned node states are encoded as: 0 = susceptible/inactive,
    1 = active/infected.

    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``edge_index`` (the edge set :math:`E`)
        and ``num_nodes`` (:math:`|V|`).
    :type data: torch_geometric.data.Data
    :param seeds: Nodes whose initial state is *active*.  Pass a list of
        node IDs, or a float in (0, 1) to activate that fraction of nodes
        chosen uniformly at random.
    :type seeds: list[int] | float
    :param threshold: Per-edge activation probability.  When
        ``threshold > 0``, every edge uses the same probability
        :math:`p=\text{threshold}`.  When ``threshold == 0``, the model
        assigns edge-specific probabilities :math:`p_{ji}=1/\deg_{in}(i)`.
    :type threshold: float
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
                 threshold: float,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         threshold=threshold)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        if self.threshold == 0:
            num_nodes = self.data.num_nodes
            edge_index = self.data.edge_index
            in_deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
            self.node_status['edge_threshold'] = 1.0 / in_deg[edge_index[1]]
        else:
            self.node_status['edge_threshold'] = None

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        self.model = IC_process(self.data.edge_index, self.threshold)



    def run_iteration(self):
        """Execute a single simulation step.

        The internal ``node_status`` is updated so that subsequent calls continue from the latest state.

        :return: Node states after one step, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_iterations(1)


    def run_iterations(self, times):
        """Execute *times* simulation steps sequentially.

        The internal ``node_status`` is updated in-place so that subsequent
        calls continue from the latest state.

        :param times: Number of steps to run.
        :type times: int
        :return: Node states at final step, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self.model._set_iterations(times)
        out_all = self.model(self.node_status)
        self.node_status['SI'], self.node_status['R_mask'] = out_all[0].squeeze(0), out_all[1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self, iterations_times=0):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node states are **re-initialised** before the epoch starts.

        :param iterations_times: Number of simulation steps per epoch.
            If ``0``, run until no node remains newly active.
        :type iterations_times: int
        :return: Node states at final step of the epoch, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times=0, batch_size=1):
        """Run multiple independent Monte-Carlo epochs in batches.

        Node states are **re-initialised** before the run.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param iterations_times: Number of simulation steps per epoch.
            If ``0``, each epoch runs until no node remains newly active.
        :type iterations_times: int
        :param batch_size: *(optional)* Number of epochs processed
            in parallel per batch.
            Defaults to ``1``.
        :type batch_size: int
        :return: Node states at final step of all epochs, shape ``(epochs, N)``.
        :rtype: torch.Tensor
        """
        try:
            check_int(epochs=epochs, iterations_times=iterations_times, batch_size=batch_size)
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
        out, R_mask = out_all
        final = out.float()
        final[R_mask] = 1
        return final.squeeze(-1)

class IC_process(Diffusion_process):
    def __init__(self, edge_index, threshold):
        super().__init__(edge_index=edge_index,
                         threshold=threshold)

    def forward(self, node_status, epochs=1):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1)
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        if node_status['edge_threshold'] is not None:
            self.threshold = node_status['edge_threshold'].unsqueeze(0).unsqueeze(2)
        while self.iterations_times > self.times or not self.iterations_times:

            temp = self.propagate(self.edge_index, x=x.float())*~R_mask
            I_p = 1-torch.exp(temp)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)

            mask_i = (i_rand_p < I_p)

            mask = (x == True)
            R_mask[mask] = True
            x[mask_i] = True
            x[mask] = False
            self.times += 1

            if not x.any():
                break
        return x, R_mask
    def message(self, x_j):
        return torch.log(1 - self.threshold * x_j)
