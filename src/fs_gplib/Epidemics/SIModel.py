import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SIModel(DiffusionModel):
    """SI (Susceptible-Infected) diffusion model on static graphs.

    Each node starts as susceptible or infected (seed).  At every step each
    infected neighbor independently transmits the disease with probability
    ``infection_beta``.  Once infected a node stays infected permanently.

    :param data: PyTorch Geometric ``Data`` object describing the graph.
    :type data: torch_geometric.data.Data
    :param seeds: Initial infected node IDs, or a float in (0, 1) interpreted
        as the fraction of nodes chosen uniformly at random.
    :type seeds: list[int] | float
    :param infection_beta: Per-contact infection probability in [0, 1].
    :type infection_beta: float
    :param device: ``'cpu'`` or a CUDA device index.
    :type device: str | int
    :param use_weight: If ``True``, scale transmission by ``edge_attr``.
    :type use_weight: bool
    :param rand_seed: Random seed used when *seeds* is a float.
    :type rand_seed: int | None
    """

    def __init__(self,
                 data,
                 seeds,
                 infection_beta: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_beta=infection_beta)


    def _init_node_status(self):
        self.node_status = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = SI_process(self.data.edge_index, self.infection_beta,  self.data.edge_attr)
        else:
            self.model = SI_process(self.data.edge_index, self.infection_beta, None)

    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


    def run_iteration(self):
        """Execute a single simulation step.

        Equivalent to ``run_iterations(1)``.  The internal ``node_status`` is
        updated so that subsequent calls continue from the latest state.

        :return: Node states after one step.  ``1.0`` = infected,
            ``0.0`` = susceptible.
        :rtype: torch.Tensor of shape ``(N,)``
        """
        return self.run_iterations(1)


    def run_iterations(self, times: int):
        """Execute *times* simulation steps sequentially.

        The internal ``node_status`` is updated in-place so that subsequent
        calls continue from the latest state.

        :param times: Number of steps to run.
        :type times: int
        :return: Node states at each step.
        :rtype: torch.Tensor of shape ``(times, N)``
        """
        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self.model._set_iterations(times)
        out_all = self.model(self.node_status)
        self.node_status = out_all.squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self, iterations_times):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Equivalent to ``run_epochs(1, iterations_times, 1)``.  Node states are
        **re-initialised** before the epoch starts.

        :param iterations_times: Number of simulation steps per epoch.
        :type iterations_times: int
        :return: Node states at each step of the epoch.
        :rtype: torch.Tensor of shape ``(iterations_times, N)``
        """
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs: int, iterations_times: int, batch_size: int=200):
        """Run multiple independent Monte-Carlo epochs in batches.

        Node states are **re-initialised** before the run.  Results are
        collected on CPU.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param iterations_times: Number of simulation steps per epoch.
        :type iterations_times: int
        :param batch_size: Maximum number of epochs processed in parallel per
            batch (controls GPU memory usage).
        :type batch_size: int
        :return: Node states across all epochs and steps.
        :rtype: torch.Tensor of shape ``(epochs, iterations_times, N)``
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
                finals.append(final.to('cpu'))
            finals = torch.cat(finals, dim=0)
        return finals
    def _return_final(self, out_all):

        final = out_all.float()
        return final.squeeze(-1)

# SI based on Message Passing
class SI_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_beta,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(edge_index=edge_index,
                         infection_beta=infection_beta,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)


    def forward(self, node_status, epochs=1):
        x = node_status.unsqueeze(0).repeat(epochs, 1, 1)

        while self.iterations_times > self.times:
            temp = self.propagate(self.edge_index, x=x.float())
            I_p = 1 - torch.exp(temp)
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (i_rand_p < I_p)
            x[mask_i] = True

            self.times += 1
        return x
    def message(self, x_j):

        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)
