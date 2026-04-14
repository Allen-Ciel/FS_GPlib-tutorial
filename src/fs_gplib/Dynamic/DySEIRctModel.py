
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class DySEIRctModel(DiffusionModel):
    r"""Dynamic SEIR with continuous-time hazards (DySEIR_ct) on time-varying networks.

    This dynamic network model runs SEIR diffusion on a snapshot sequence
    :math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`.  The infection mechanism for
    :math:`S\to E` follows the (discrete) DySEIR model: infectious neighbors in
    the **current** snapshot expose a susceptible node with probability
    :math:`\beta` per contact (optionally scaled by snapshot edge weights).

    Unlike DySEIR, transitions :math:`E\to I` and :math:`I\to R` are governed by
    **elapsed-time dependent** probabilities.  If a node entered state :math:`E`
    at iteration :math:`t_i^E` and state :math:`I` at iteration :math:`t_i^I`,
    then at snapshot index :math:`k`:

    - :math:`P(E\to I)=1-\exp(-\alpha\,(k-t_i^E))`
    - :math:`P(I\to R)=1-\exp(-\gamma\,(k-t_i^I))`

    Returned tensors encode states as **float** values: susceptible ``0``, 
    infected ``1``, exposed ``2``, recovered ``3``.

    The number of simulation steps cannot exceed ``len(edge_index_list)``.

    :param x: Node tensor of shape ``(N, 1)``.
    :type x: torch.Tensor
    :param edge_index_list: List of snapshot ``edge_index`` tensors, length :math:`T`.
    :type edge_index_list: list[torch.Tensor]
    :param seeds: Initially infectious nodes: list of node IDs or a float in ``(0,1)``.
    :type seeds: list[int] | float
    :param infection_beta: Exposure probability :math:`\beta` (S→E), in ``[0,1]``.
    :type infection_beta: float
    :param latent_alpha: Hazard parameter :math:`\alpha > 0` for E→I.
    :type latent_alpha: float
    :param removal_gamma: Hazard parameter :math:`\gamma > 0` for I→R.
    :type removal_gamma: float
    :param device: *(optional)* ``'cpu'`` or a CUDA device index. Defaults to ``'cpu'``.
    :type device: str | int
    :param rand_seed: *(optional)* Random seed used when *seeds* is a float. Defaults to ``None``.
    :type rand_seed: int | None
    :param edge_attr_list: *(optional)* Snapshot edge weights aligned with *edge_index_list*.
    :type edge_attr_list: list[torch.Tensor] | None
    """

    def __init__(self,
                 x,
                 edge_index_list,
                 seeds,
                 infection_beta: float,
                 removal_gamma: float,
                 latent_alpha: float,
                 device='cpu',
                 rand_seed=None,
                 edge_attr_list=None):
        super().__init__(x=x,
                         edge_index_list=edge_index_list,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha,
                         edge_attr_list=edge_attr_list)

    def _validate_parameters(self, kwargs):
        # infection_beta ∈ [0, 1]; removal_gamma > 0; latent_alpha > 0
        check_float_parameter(0, 1, True, True, infection_beta=kwargs['infection_beta'])
        check_float_parameter(0, float("inf"), False, True, removal_gamma=kwargs["removal_gamma"])
        check_float_parameter(0, float("inf"), False, True, latent_alpha=kwargs["latent_alpha"])

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _init_node_status(self):
        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.x.shape[0], self.seeds).bool().to(self.device)
        self.node_status['E_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['R_mask'] = torch.zeros_like(self.node_status['SI'], dtype=torch.bool).to(self.device)
        self.node_status['E_iteration'] = torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)
        self.node_status['I_iteration'] = torch.zeros_like(self.node_status['SI'], dtype=torch.int32).to(self.device)

    def _set_device(self, device):
        super()._set_device(device)
        self._init_node_status()

        self.model = DySEIRctModel_process(self.edge_index_list, self.edge_attr_list, self.infection_beta, self.removal_gamma, self.latent_alpha).to(self.device)

    # def _set_seed(self, seeds):
    #     super()._initialize_seeds(seeds)
    #     self._init_node_status()


    def run_iteration(self):
        """Advance the epidemic by one snapshot step.

        The internal ``node_status`` is updated so that subsequent calls continue from the latest
        state. Requires at least one remaining snapshot.

        :return: Node states after that step, shape ``(1, 1, N)`` (values ``0``–``3``).
        :rtype: torch.Tensor
        """
        return self.run_iterations(1)


    def run_iterations(self, times):
        """Run *times* consecutive snapshot steps on the evolving graph sequence.

        The internal ``node_status`` is updated to the state after the last step. Requires
        ``len(edge_index_list) - t >= times`` where :math:`t` is the number of steps already
        consumed on this process.

        :param times: Number of snapshots to advance (must not exceed remaining snapshots).
        :type times: int
        :return: Node states after each step, stacked with shape ``(times, 1, N)`` (values
            ``0``–``3``).
        :rtype: torch.Tensor
        """
        check_int(times=times)

        if len(self.edge_index_list) - self.model.times < times:
            raise ValueError('The number of remaining snapshots must be larger than iteration times')
        x_list, R_mask_list, E_mask_list, E_iteration_list, I_iteration_list = self.model(self.node_status, iterations_times=times)

        out_all = torch.stack(x_list, dim=0), torch.stack(R_mask_list, dim=0), torch.stack(E_mask_list, dim=0), torch.stack(E_iteration_list, dim=0), torch.stack(I_iteration_list, dim=0)

        self.node_status['SI'], \
            self.node_status['R_mask'], \
            self.node_status['E_mask'], \
            self.node_status['E_iteration'], \
            self.node_status['I_iteration']\
            = out_all[0][-1].squeeze(0), \
            out_all[1][-1].squeeze(0), \
            out_all[2][-1].squeeze(0), \
            out_all[3][-1].squeeze(0), \
            out_all[4][-1].squeeze(0)
        final = self._return_final(out_all)
        return final

    def run_epoch(self):
        """Run one Monte-Carlo realisation over the **full** snapshot sequence.

        The process internal step counter is reset; node states are **re-initialised** before the
        epoch starts.

        :return: Node states trajectory over all snapshots, shape ``(T, 1, N)`` with
            :math:`T =` ``len(edge_index_list)`` (values ``0``–``3``).
        :rtype: torch.Tensor
        """
        return self.run_epochs(1, 1)

    def run_epochs(self, epochs, batch_size=200):
        """Run multiple independent Monte-Carlo realisations in batches.

        For each realisation the snapshot index is reset to the beginning and the epidemic is
        evolved through **all** snapshots. Node states are **re-initialised** before the run.

        :param epochs: Total number of independent realisations.
        :type epochs: int
        :param batch_size: *(optional)* Parallel epochs per batch. Defaults to ``200``.
        :type batch_size: int
        :return: Node states trajectories for all realisations, shape ``(T, E, N)`` where
            :math:`T =` ``len(edge_index_list)`` and :math:`E` is *epochs* (values ``0``–``3``).
        :rtype: torch.Tensor
        """

        check_int(epochs=epochs, batch_size=batch_size)

        self._init_node_status()
        epoch_groups = epochs_groups_list(epochs, batch_size)
        bar = tqdm(epoch_groups)
        finals = []

        with torch.no_grad():
            for i, epoch_group in enumerate(bar):
                bar.set_description('Batch {}'.format(i))
                self.model._set_iterations()
                out_list, R_mask_list, E_mask_list, _, _ = self.model(self.node_status, epoch_group)

                out_all = torch.stack(out_list, dim=0), torch.stack(R_mask_list, dim=0), torch.stack(E_mask_list, dim=0), None, None
                final = self._return_final(out_all)
                final_cpu = final.to('cpu')
                finals.append(final_cpu)
            finals = torch.cat(finals, dim=1)
        return finals

    def _return_final(self, out_all):
        out, R_mask, E_mask, _, _ = out_all
        final = out.float()
        final[R_mask == True] = 3
        final[E_mask == True] = 2
        return final.squeeze(-1)

class DySEIRctModel_process(Diffusion_process):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 infection_beta,
                 removal_gamma,
                 latent_alpha):
        super().__init__(edge_index_list=edge_index_list,
                         edge_attr_list=edge_attr_list,
                         infection_beta=infection_beta,
                         removal_gamma=removal_gamma,
                         latent_alpha=latent_alpha)

    def forward(self, node_status, epochs=1, iterations_times = None):
        x = node_status['SI'].unsqueeze(0).repeat(epochs, 1, 1) # [k, N, 1]
        R_mask = node_status['R_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        E_mask = node_status['E_mask'].unsqueeze(0).repeat(epochs, 1, 1)
        E_iteration = node_status['E_iteration'].unsqueeze(0).repeat(epochs, 1, 1)
        I_iteration = node_status['I_iteration'].unsqueeze(0).repeat(epochs, 1, 1)

        x_list = []
        R_mask_list = []
        E_mask_list = []
        E_iteration_list = []
        I_iteration_list = []
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

            # S→E
            temp = self.propagate(edge_index, x=(x & ~E_mask).float()) * ~R_mask
            E_p = 1 - torch.exp(temp)
            e_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_e = (e_rand_p < E_p) & (~x & ~R_mask)
            E_iteration[mask_e] = self.times

            # E→I
            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            I_p = 1 - torch.exp(-(self.times - E_iteration) * self.latent_alpha)
            mask_i = E_mask & (i_rand_p < I_p)
            I_iteration[mask_i] = self.times

            # I→R
            r_rand_p = torch.rand_like(x, dtype=torch.float32)
            R_p = 1 - torch.exp(-(self.times - I_iteration) * self.removal_gamma)
            mask_r = (x & ~E_mask) & (r_rand_p < R_p)

            x[mask_r] = False
            x[mask_e] = True
            E_mask[mask_e] = True
            E_mask[mask_i] = False
            R_mask[mask_r] = True

            self.times += 1
            x_list.append(x.clone())
            R_mask_list.append(R_mask.clone())
            E_mask_list.append(E_mask.clone())
            E_iteration_list.append(E_iteration.clone())
            I_iteration_list.append(I_iteration.clone())

        return x_list, R_mask_list, E_mask_list, E_iteration_list, I_iteration_list

    def message(self, x_j):
        return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

