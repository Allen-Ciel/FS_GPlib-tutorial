import sys
from tqdm import tqdm

from .base import DiffusionModel, Diffusion_process
from ..utils import *

class ProfileModel(DiffusionModel):
    r"""Profile diffusion model on static graphs.

    This model describes social contagion with node-specific susceptibility.
    Each node starts as inactive or active (seed).  At every step a
    non-blocked inactive node may first become active spontaneously with
    probability :math:`\alpha`.  If it is exposed to at least one active
    neighbor and does not adopt spontaneously, it adopts with probability
    equal to its profile value.  If exposed but still not adopted, it may
    become blocked with probability :math:`\beta` and then remain permanently
    ineligible for activation.  Once activated, a node stays active.

    Returned node states are encoded as: -1 = blocked, 0 = inactive,
    1 = active.


    :param data: PyTorch Geometric ``Data`` object representing graph
        :math:`G=(V,E)`.  Must contain ``edge_index`` (the edge set :math:`E`)
        and ``num_nodes`` (:math:`|V|`).
    :type data: torch_geometric.data.Data
    :param seeds: Nodes whose initial state is *Active*.  Pass a list of
        node IDs, or a float in (0, 1) to activate that fraction of nodes
        chosen uniformly at random.
    :type seeds: list[int] | float
    :param profile: Per-node adoption propensity after exposure.  When
        ``profile \in (0,1]``, all nodes share the same profile value.  When
        ``profile == 0``, a profile is sampled independently for each node
        from ``Uniform(0,1)``.
    :type profile: float
    :param adopter_rate: Per-step spontaneous adoption probability
        :math:`\alpha \in [0,1]` for each non-blocked inactive node.
    :type adopter_rate: float
    :param blocked_rate: Probability :math:`\beta \in [0,1]` that an exposed
        node becomes blocked after failing to adopt via its profile.
    :type blocked_rate: float
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
                 profile: float,
                 adopter_rate,
                 blocked_rate,
                 device='cpu',
                 rand_seed=None):
        super().__init__(data=data,
                         seeds=seeds,
                         device=device,
                         rand_seed=rand_seed,
                         profile=profile,
                         adopter_rate=adopter_rate,
                         blocked_rate=blocked_rate)


    def _init_node_status(self):

        self.node_status = dict()
        self.node_status['SI'] = get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device)
        if self.profile == 0:
            self.node_status["node_profile"] = torch.rand_like(self.node_status['SI'], dtype=torch.float32)
        else:
            self.node_status["node_profile"] = torch.full_like(self.node_status['SI'].float(), self.profile)
        self.node_status['B_mask'] = None

    def _set_device(self, device):
        super()._set_device(device)
        self.data = self.data.to(self.device)
        self._init_node_status()
        self.model = Profile_process(self.data.edge_index, self.profile, self.adopter_rate, self.blocked_rate).to(self.device)

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
        self.node_status['SI'], self.node_status['B_mask'] = out_all[0].squeeze(0), out_all[1].squeeze(0) #[N, 1]
        final = self._return_final(out_all)
        return final # [1, N]

    def run_epoch(self, iterations_times):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node states are **re-initialised** before the epoch starts.

        :param iterations_times: Number of simulation steps per epoch.
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
        :type iterations_times: int
        :param batch_size: *(optional)* Number of epochs processed
            in parallel per batch.
            Defaults to ``1``.
        :type batch_size: int
        :return: Node states at final step of all epochs, shape ``(epochs, N)``.
        :rtype: torch.Tensor
        """
        try:
            check_int(epochs=epochs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)
        self._init_node_status()

        n = int(epochs / batch_size)
        epoch_groups = [batch_size] * n
        last_epoch_group = epochs - n * batch_size
        if last_epoch_group != 0:
            epoch_groups.append(last_epoch_group)
        bar = tqdm(epoch_groups)
        finals = []
        with torch.no_grad():
            for i, epoch_group in enumerate(bar):
                bar.set_description('Batch {}'.format(i))
                self.model._set_iterations(iterations_times)
                self.node_status['B_mask'] = None
                out_all = self.model(self.node_status, epoch_group)
                final = self._return_final(out_all)
                finals.append(final.to('cpu'))
            finals = torch.cat(finals, dim=0)
        return finals

    def _return_final(self, out_all):
        out, B_mask = out_all
        final = out.float()
        final[B_mask] = -1
        return final.squeeze(-1)

class Profile_process(Diffusion_process):
    def __init__(self,
                 edge_index,
                 profile,
                 adopter_rate,
                 blocked_rate):
        super().__init__(edge_index=edge_index,
                         profile=profile,
                         adopter_rate=adopter_rate,
                         blocked_rate=blocked_rate)

    def forward(self, node_status, epochs=1):
        x = node_status['SI'].repeat(epochs, 1, 1)
        if node_status['B_mask'] is None:
            B_mask = torch.zeros_like(x, dtype=torch.bool).expand(epochs, -1, -1)
        else:
            B_mask = node_status['B_mask'].expand(epochs, -1, -1)
        node_profile = node_status["node_profile"].expand(epochs, -1, -1)
        while self.iterations_times > self.times:

            ai_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_ai = (~x & ~B_mask) & (ai_rand_p < self.adopter_rate)


            I_p = self.propagate(self.edge_index, x=x)*~B_mask

            i_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_i = I_p & (~x & ~B_mask) & (ai_rand_p >= self.adopter_rate) & (node_profile > i_rand_p) #(I_p & ~mask_ai & ~x) & (node_profile > i_rand_p)

            b_rand_p = torch.rand_like(x, dtype=torch.float32)
            mask_b = I_p & (~x & ~B_mask) & (ai_rand_p >= self.adopter_rate) & (node_profile <= i_rand_p) & (self.blocked_rate > b_rand_p) #(I_p & ~mask_ai & ~x) & (node_profile <= i_rand_p) & (self.blocked_rate > b_rand_p)

            x[mask_i] = True
            B_mask[mask_b] = True
            x[mask_ai] = True

            self.times += 1

        return x, B_mask




