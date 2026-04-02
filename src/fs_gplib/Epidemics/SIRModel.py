import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SIRModel(DiffusionModel):
    r"""SIR (Susceptible-Infected-Recovered) diffusion model on static graphs.

    Each node starts as susceptible or infected (seed).  At every step each
    infected neighbor independently transmits the disease with probability
    :math:`\beta`, while infected nodes recover with probability
    :math:`\gamma` and become permanently immune.

    Returned node states are encoded as: 0 = susceptible, 1 = infected,
    2 = recovered.

    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``edge_index`` (the edge set :math:`E`)
        and ``num_nodes`` (:math:`|V|`).  When *use_weight* is ``True``,
        ``edge_attr`` supplies per-edge weights :math:`w_{ji}`.
    :type data: torch_geometric.data.Data
    :param seeds: Nodes whose initial state is *Infected*.  Pass a list
        of node IDs, or a float in (0, 1) to infect that fraction of
        nodes chosen uniformly at random.
    :type seeds: list[int] | float
    :param infection_beta: Per-contact infection probability
        :math:`\beta \in [0,1]`.
    :type infection_beta: float
    :param recovery_lambda: Per-step recovery probability
        :math:`\gamma \in [0,1]`.
    :type recovery_lambda: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param use_weight: *(optional)* If ``True``, each edge :math:`(j,i)`
        carries a weight :math:`w_{ji}` from ``data.edge_attr`` and the
        infection probability becomes :math:`\beta w_{ji}`.  If ``False``
        all weights default to 1 (i.e. :math:`w_{ji}=1`).
        Defaults to ``False``.
    :type use_weight: bool
    :param rand_seed: *(optional)* Random seed used when *seeds* is a
        float.  Defaults to ``None``.
    :type rand_seed: int | None
    """

    def __init__(self,
                 data,
                 seeds,
                 infection_beta: float,
                 recovery_lambda: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_beta=infection_beta,
                         recovery_lambda=recovery_lambda)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)

        self._init_node_status()
        if self.use_weight:
            self.model = SIR_process(self.data.edge_index, self.infection_beta, self.recovery_lambda,self.data.edge_attr).to(self.device)
        else:
            self.model = SIR_process(self.data.edge_index, self.infection_beta, self.recovery_lambda, None).to(self.device)

    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


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

    def run_epoch(self, iterations_times):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node states are **re-initialised** before the epoch starts.

        :param iterations_times: Number of simulation steps per epoch.
        :type iterations_times: int
        :return: Node states at final step of the epoch, shape ``(1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, iterations_times, 1)

    def run_epochs(self, epochs, iterations_times, batch_size=200):
        """Run multiple independent Monte-Carlo epochs in batches.

        Node states are **re-initialised** before the run.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param iterations_times: Number of simulation steps per epoch.
        :type iterations_times: int
        :param batch_size: *(optional)* Number of epochs processed
            in parallel per batch.
            Defaults to ``200``.
        :type batch_size: int
        :return: Node states at final step of all epochs, shape ``(epochs, N)``.
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
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=0)

        return finals

    def _return_final(self, out_all):
        out, R_mask = out_all
        final = out.float()
        final[R_mask == True] = 2
        return final.squeeze(-1)

class SIR_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_beta,
                 recovery_lambda,
                 # iterations_times,
                 edge_attr=None):
        super().__init__(
                         edge_index=edge_index,
                         infection_beta=infection_beta,
                         recovery_lambda=recovery_lambda,
                         # iterations_times=iterations_times,
                         edge_attr=edge_attr)

    def forward(self, node_status, epochs=1):
        # self.epochs = epochs
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)

        while self.iterations_times > self.times:

            temp = self.propagate(self.edge_index, x=x.float()) * ~R_mask
            I_p = 1-torch.exp(temp)

            i_rand_p = torch.rand_like(x.float(), dtype=torch.float32)
            mask_i = (i_rand_p < I_p)

            e_rand_p = torch.rand_like(x.float(), dtype=torch.float32)
            mask = (x == True) & (e_rand_p < self.recovery_lambda)
            R_mask[mask] = True

            x[mask_i] = True
            x[mask] = False
            self.times += 1
        return x, R_mask

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)


