import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DySIModel(DiffusionModel):
    r"""Dynamic SI (DySI) epidemic model on **time-varying networks** (dynamic network models).

    This model extends the classical static SI dynamics (see the epidemic ``SIModel``) to a
    sequence of graph snapshots :math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`: at step :math:`k`
    infections propagate only along edges present in :math:`E^{(k)}`.  Each infected neighbor
    transmits with probability :math:`\beta` (optionally scaled by per-edge weights in
    *edge_attr_list*); once infected, a node stays infected.

    The number of simulation steps cannot exceed the number of snapshots ``len(edge_index_list)``.
    Supply an explicit node tensor *x* (shape ``(N, 1)``) and lists *edge_index_list* /
    *edge_attr_list* instead of a single PyG ``Data`` object.

    :param x: Node feature tensor of shape ``(N, 1)`` (node count :math:`N` is inferred from the
        leading dimension).
    :type x: torch.Tensor
    :param edge_index_list: One ``edge_index`` tensor per snapshot, length :math:`T`, defining
        :math:`E^{(k)}` at each step.
    :type edge_index_list: list[torch.Tensor]
    :param seeds: Initially infected nodes: a list of integer node IDs, or a float in ``[0, 1)``
        to infect that fraction of nodes chosen uniformly at random.
    :type seeds: list[int] | float
    :param infection_beta: Per-contact infection probability :math:`\beta \in [0, 1]`.
    :type infection_beta: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param rand_seed: *(optional)* Random seed used when *seeds* is a float.
        Defaults to ``None``.
    :type rand_seed: int | None
    :param edge_attr_list: *(optional)* One edge-weight tensor per snapshot, aligned with
        *edge_index_list*.  If ``None``, all edge weights are treated as ``1``.
    :type edge_attr_list: list[torch.Tensor] | None
    """

    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 infection_beta: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         infection_beta=infection_beta,
                         edge_attr_list=edge_attr_list)

    def _init_node_status(self):
        self.node_status = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()
        self.model = DySIModel_process(self.edge_index_list, self.edge_attr_list, self.infection_beta).to(self.device)

    def _set_seed(self, seeds):
        super()._initialize_seeds(seeds)
        self._init_node_status()


    def run_iteration(self):
        """Advance the epidemic by one snapshot step.

        The internal ``node_status`` is updated so that subsequent calls continue from the latest
        state.  Requires at least one remaining snapshot.

        :return: State tensor after that step, shape ``(1, 1, N)``.
        :rtype: torch.Tensor
        """
        return self.run_iterations(1)


    def run_iterations(self, times):
        """Run *times* consecutive snapshot steps on the evolving graph sequence.

        The internal ``node_status`` is updated to the state after the last step.  Requires
        ``len(edge_index_list) - t >= times`` where :math:`t` is the number of steps already
        consumed on this process.

        :param times: Number of snapshots to advance (must not exceed remaining snapshots).
        :type times: int
        :return: States after each of the *times* steps, stacked with shape ``(times, 1, N)``.
        :rtype: torch.Tensor
        """
        try:
            check_int(times=times)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)
        if len(self.edge_index_list) - self.model.times < times:
            raise ValueError('The number of remaining snapshots must be larger than iteration times')
        x_list = self.model(self.node_status, iterations_times=times)
        out_all = torch.stack(x_list, dim=0)
        self.node_status = out_all[-1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self):
        """Run one Monte-Carlo realisation over the **full** snapshot sequence.

        The process internal step counter is reset; node states are **re-initialised** before the epoch starts.

        :return: State trajectory over all snapshots, shape ``(T, 1, N)`` with
            :math:`T =` ``len(edge_index_list)``.
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, 1)

    def run_epochs(self, epochs, batch_size=200):
        """Run multiple independent Monte-Carlo realisations in batches.

        For each realisation the snapshot index is reset to the beginning and the epidemic is
        evolved through **all** snapshots.  Node states are **re-initialised** before the run.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param batch_size: *(optional)* Parallel epochs per batch.
            Defaults to ``200``.
        :type batch_size: int
        :return: Trajectories for all realisations, shape ``(T, E, N)`` where
            :math:`T =` ``len(edge_index_list)`` and :math:`E` is *epochs*.
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

class DySIModel_process(Diffusion_process):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 infection_beta):
        super().__init__(edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         infection_beta=infection_beta)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status.unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        x_list = []

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

            temp = self.propagate(edge_index, x=x.float())
            I_p = 1-torch.exp(temp)

            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = (i_rand_p < I_p)
            x[mask_i] = True

            self.times += 1
            x_list.append(x.clone())

        return x_list

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

