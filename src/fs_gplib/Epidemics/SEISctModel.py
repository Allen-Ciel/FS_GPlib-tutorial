import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class SEISctModel(DiffusionModel):
    r"""SEIS_ct (continuous-time SEIS) diffusion model on static graphs.

    Variant of SEIS where the E→I and I→S transition probabilities depend on
    the elapsed time since entering that state, rather than being constant per
    step.  Specifically, :math:`P(E \to I) = 1 - e^{-\Delta t^E \,\alpha}`
    and :math:`P(I \to S) = 1 - e^{-\Delta t^I \,\gamma}`.

    Returned node states are encoded as: 0 = susceptible, 1 = infected,
    2 = exposed.

    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``edge_index`` (the edge set :math:`E`)
        and ``num_nodes`` (:math:`|V|`).  When *use_weight* is ``True``,
        ``edge_attr`` supplies per-edge weights :math:`w_{ji}`.
    :type data: torch_geometric.data.Data
    :param seeds: Nodes whose initial state is *Infected*.  Pass a list
        of node IDs, or a float in (0, 1) to infect that fraction of
        nodes chosen uniformly at random.
    :type seeds: list[int] | float
    :param infection_beta: Per-contact exposure probability
        :math:`\beta \in [0,1]` (S→E).
    :type infection_beta: float
    :param removal_gamma: Recovery rate :math:`\gamma \in [0,1]` (I→S).
        Used in :math:`1 - e^{-\Delta t^I \,\gamma}`.
    :type removal_gamma: float
    :param latent_alpha: Incubation rate :math:`\alpha \in [0,1]` (E→I).
        Used in :math:`1 - e^{-\Delta t^E \,\alpha}`.
    :type latent_alpha: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param use_weight: *(optional)* If ``True``, each edge :math:`(j,i)`
        carries a weight :math:`w_{ji}` from ``data.edge_attr`` and the
        exposure probability becomes :math:`\beta w_{ji}`.  If ``False``
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
                 removal_gamma: float,
                 latent_alpha: float,
                 device='cpu',
                 use_weight: bool = False,
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         use_weight=use_weight,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha)


    def _init_node_status(self):
        self.node_status = dict()
        # mask
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        self.node_status['E_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)

        self.node_status['E_iteration'] = torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)
        self.node_status['I_iteration'] = torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        if self.use_weight:
            self.model = SEISct_process(self.data.edge_index, self.infection_beta, self.removal_gamma, self.latent_alpha, 0, self.data.edge_attr)
        else:
            self.model = SEISct_process(self.data.edge_index, self.infection_beta, self.removal_gamma,
                                        self.latent_alpha, 0, None)

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
        self.node_status['SI'], self.node_status['E_mask'], self.node_status['E_iteration'], self.node_status['I_iteration'] = out_all[0].squeeze(0), out_all[1].squeeze(0), out_all[2].squeeze(0), out_all[3].squeeze(0)
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
                finals.append(final.to('cpu'))
        finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        out, E_mask, _, _ = out_all
        final = out.float()
        final[E_mask == True] = 2
        return final.squeeze(-1)

class SEISct_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 infection_beta,
                 removal_gamma,
                 latent_alpha,
                 iterations_times,
                 edge_attr=None):
        super().__init__(edge_index=edge_index,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha,
                         iterations_times=iterations_times,
                         edge_attr=edge_attr) # 设置聚合方式为求和


    def forward(self, node_status, epochs=1):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1)

        E_mask = node_status['E_mask'].unsqueeze(0).repeat(epochs, 1, 1)

        E_iteration = node_status['E_iteration'].unsqueeze(0).repeat(epochs, 1, 1)
        I_iteration = node_status['I_iteration'].unsqueeze(0).repeat(epochs, 1, 1)

        while self.times < self.iterations_times:


            # S to E
            temp = self.propagate(self.edge_index, x=(x & ~E_mask).float())
            E_p = 1 - torch.exp(temp)
            e_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_e = (e_rand_p < E_p) & (~x)
            E_iteration[mask_e] = self.times

            # E to I
            i_rand_p = torch.rand_like(E_mask.float())
            I_p = 1 - torch.exp(-(self.times - E_iteration) * self.latent_alpha)
            mask_i = E_mask & (i_rand_p < I_p)
            I_iteration[mask_i] = self.times

            # I to S
            s2_rand_p = torch.rand_like(x, dtype=torch.float32)
            S2_p = 1 - torch.exp(-(self.times - I_iteration) * self.removal_gamma)
            mask_s2 = (x & ~E_mask) & (s2_rand_p < S2_p)

            x[mask_e] = True
            x[mask_s2] = False
            E_mask[mask_e] = True
            E_mask[mask_i] = False
            self.times += 1


        return x, E_mask, E_iteration, I_iteration

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)
