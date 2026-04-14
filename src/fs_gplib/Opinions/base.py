import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from tqdm import tqdm

from ..utils import *

class DiffusionModel:
    """Base class for all opinion dynamics models on static graphs.

    ``DiffusionModel`` (in the Opinions package) provides the shared
    initialisation logic and a uniform simulation interface that every
    concrete opinion model (Voter, QVoter, MajorityRule, Sznajd, HK, WHK)
    inherits.  Users typically do **not** instantiate this class directly;
    instead they create a subclass such as
    :class:`~fs_gplib.Opinions.VoterModel.VoterModel`.

    **Initialisation pipeline** (executed in ``__init__``):

    1. Validate the input graph via :meth:`_validate_graph`.
    2. Initialise seed nodes (or initial opinions) via
       :meth:`_initialize_seeds`.
    3. Validate and store model-specific parameters via
       :meth:`_validate_parameters`.
    4. Transfer the model to the target device via :meth:`_set_device`.

    **Simulation interface** – four progressively higher-level entry points:

    +-----------------------------------------+-----------------------------------------------------------+
    | Method                                  | Description                                               |
    +=========================================+===========================================================+
    | :meth:`run_iteration`                   | Advance **one** time step from the current state.         |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_iterations(times)            | Advance *times* steps from the current state.             |
    | <run_iteration>`                        |                                                           |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_epoch(times)                 | Reset to initial state, then run *times* steps            |
    | <run_epoch>`                            | (one independent realisation).                            |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_epochs(epochs, times,        | Run *epochs* independent realisations in batches          |
    | batch_size) <run_epochs>`               | (Monte-Carlo simulation).                                 |
    +-----------------------------------------+-----------------------------------------------------------+

    :param data: PyTorch Geometric ``Data`` object representing the graph.
        Must contain ``edge_index``.
    :type data: torch_geometric.data.Data
    :param seeds: Initial opinion configuration.

        * For **discrete** opinion models (e.g. Voter): a **list of node IDs**
          holding opinion ``1``, or a **float in [0, 1)** representing the
          fraction of nodes to set to opinion ``1`` uniformly at random.
          ``None`` is accepted for models that set their own initial state.
        * For **continuous** opinion models (e.g. HK): a **list of floats**
          giving the initial opinion of every node, or ``None`` to sample
          uniformly from :math:`(-1, 1)`.
    :type seeds: float | list[int] | list[float] | None
    :param rand_seed: Random seed for NumPy / Python RNG, used when *seeds*
        is a float or ``None`` to make the random initialisation
        reproducible.  Defaults to ``None``.
    :type rand_seed: int | None
    :param device: ``'cpu'`` or a CUDA device index (e.g. ``0``).
        Defaults to ``'cpu'``.
    :type device: str | int
    :param kwargs: Model-specific parameters (e.g. ``epsilon`` for HK).
        Each key-value pair is validated and stored as an instance attribute.
    """

    def __init__(self, data, seeds: Union[float, List[int], None], rand_seed=None, device='cpu', **kwargs):
        self._skip_init = False
        self.rand_seed = rand_seed

        self._validate_graph(data)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)

    def _validate_graph(self, data):
        """Validate the input graph.

        Checks that *data* is a PyG ``Data`` object and records whether the
        graph is directed (stored in ``self.is_directed``).

        :param data: The input graph.
        :type data: torch_geometric.data.Data
        :raises ValueError: If *data* is not a ``Data`` object.
        """
        if not isinstance(data, Data):
            raise ValueError("data must be a Data object from the PyG library.")
        self.data = data

        self.is_directed = self.data.is_directed()

    def _get_num_nodes(self, data):
        """Return the number of nodes in *data*.

        Tries ``data.num_nodes`` first, then falls back to
        ``data.x.size(0)``.

        :param data: The input graph.
        :type data: torch_geometric.data.Data
        :return: Number of nodes :math:`|V|`.
        :rtype: int
        :raises ValueError: If the node count cannot be determined.
        """
        if hasattr(data, 'num_nodes'):
            return data.num_nodes
        elif hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            return data.x.size(0)
        else:
            raise ValueError("The number of nodes in data cannot be determined.")

    def _initialize_seeds(self, seeds):
        """Parse *seeds* and store the result in ``self.seeds``.

        This low-level method only updates ``self.seeds`` without refreshing
        node-state tensors.  For post-construction use, prefer
        :meth:`_set_seed` which also calls :meth:`_init_node_status`
        automatically.

        For **discrete** opinion models the semantics match the epidemic base
        class: a float is interpreted as a fraction, and a list gives explicit
        node IDs with opinion ``1``.  **Continuous** opinion models (e.g. HK)
        override this method to accept a list of floats instead.

        :param seeds: Initial opinion configuration.

            * **float in (0, 1)** – fraction of nodes to set to opinion ``1``
              (selected uniformly at random using ``rand_seed``).
            * **list[int]** – explicit node indices holding opinion ``1``.
            * **None** – no initial configuration (model sets its own).
        :type seeds: float | list[int] | None
        :raises ValueError: If *seeds* is a float outside [0, 1), the list
            contains non-integer elements, or indices exceed graph size.
        """

        self.num_nodes = self._get_num_nodes(self.data)

        if isinstance(seeds, (float, int)):
            if seeds == True or seeds == False:
                raise ValueError("The seeds must be a decimal number in the range (0,1) or a list of integers.")
            if not (0 < seeds < 1):
                raise ValueError("When 'seeds' is a decimal number, it must be in the range (0,1).")
            np.random.seed(self.rand_seed)
            seed_count = int(self.num_nodes * seeds)
            self.seeds = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
        elif isinstance(seeds, list):
            # check the type of the elements in the seeds list
            if not all(isinstance(i, int) and i >= 0 for i in seeds):
                raise ValueError("When seeds are a list, the elements must be positive integers.")
            # check if the seeds list is empty
            if len(seeds) == 0:
                raise ValueError("The seeds list cannot be empty.")
            # check if the seeds list contains out of range indices
            out_of_range = [s for s in seeds if s >= self.num_nodes]
            if out_of_range:
                raise ValueError(
                    f"Seed indices {out_of_range} are out of range. "
                    f"Valid seed indices are 0 to {self.num_nodes - 1}."
                )
                #raise ValueError("The maximum value in the seeds list cannot exceed the number of nodes in data.")
            self.seeds = seeds
        elif seeds is None:
            self.seeds = seeds
        else:
            raise ValueError("seeds must be decimals in the range (0,1), a list of integers, or None.")

    def _set_device(self, device):
        """Set (or switch) the computation device for the model.

        When called during initialisation the graph data and internal tensors
        are moved to the target device.  It can also be called **after**
        construction to migrate an existing model to a different device
        (subclasses should override this method to move additional tensors).

        :param device: ``'cpu'`` or a CUDA device index (e.g. ``0``).
        :type device: str | int
        :raises Exception: If a GPU is requested but CUDA is not available.

        .. rubric:: Example

        .. code-block:: python

            model = VoterModel(data, seeds, device='cpu')
            # switch to GPU 0
            model._set_device(0)
        """
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:{}'.format(device))
            else:
                raise Exception(f"cuda is not available!")

    def _validate_parameters(self, kwargs):
        """Validate and store model-specific parameters.

        Every keyword argument is validated via :func:`check_float_parameter` and
        then stored as an instance attribute with the same name.

        :param kwargs: Model-specific parameters (e.g. ``epsilon=0.3``).
        :type kwargs: dict
        :raises SystemExit: If any value fails validation.
        """
        check_float_parameter(0, 1, True, True, **kwargs)

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _init_node_status(self):
        """Reset all node opinions to the initial configuration.

        This method rebuilds the internal ``node_status`` tensor(s) so that
        the opinion state matches the original seed configuration.

        Subclasses **must** override this method to create the status
        tensors appropriate for their opinion representation (e.g. a single
        boolean tensor for Voter, or a float tensor for HK).

        Calling this method is useful when you want to **rerun** a simulation
        from scratch without reconstructing the model object.

        .. rubric:: Example

        .. code-block:: python

            result1 = model.run_iterations(100)
            model._init_node_status()   # reset
            result2 = model.run_iterations(100)
        """
        pass

    def _set_seed(self, seeds):
        """Replace the current seed / opinion configuration and refresh node states.

        This is the recommended way to change seeds **after** construction.
        It calls :meth:`_initialize_seeds` to parse and store the new seeds,
        then (if the model is fully initialised) :meth:`_init_node_status` to
        rebuild the node-state tensors on the current device.

        :param seeds: New seed specification – same format as the constructor
            parameter *seeds* (a float fraction, a list of node IDs / opinion
            values, or ``None``).
        :type seeds: float | list[int] | list[float] | None

        .. rubric:: Example

        .. code-block:: python

            model = VoterModel(data, seeds=[0, 1, 2], ...)
            # later, switch to a random 20 % opinion-1 set
            model._set_seed(0.2)
        """
        self._initialize_seeds(seeds)
        if hasattr(self, 'device'):
            self._init_node_status()

    def run_epoch(self, **kwargs):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node opinions are **re-initialised** before the epoch starts, so
        each call produces an independent simulation trajectory.

        .. note::
            This is a placeholder in the base class.  Concrete subclasses
            provide the actual implementation.
        """
        pass

    def run_epochs(self, epochs, iterations_times):
        """Run multiple independent Monte-Carlo epochs sequentially.

        This is a simplified version used by some legacy subclasses; most
        concrete models override :meth:`run_epochs` with a batch-parallel
        implementation for better performance.

        :param epochs: Total number of independent realisations
            (Monte-Carlo runs).
        :type epochs: int
        :param iterations_times: Number of simulation steps per epoch.
        :type iterations_times: int
        :return: A list of per-epoch results.
        :rtype: list
        """
        check_int(iterations_times=iterations_times, epochs=epochs)

        self._init_node_status()

        bar = tqdm(range(epochs))
        final = []
        self._skip_init = True
        with torch.no_grad():
            for i in bar:
                bar.set_description('Batch {}'.format(i))
                out = self.run_epoch(iterations_times=iterations_times)
                final.append(out)
        return final


    def iteration(self):
        """Execute a single simulation step from the current node state.

        The internal ``node_status`` is updated in-place so that
        subsequent calls continue from the latest state.

        .. note::
            This is a placeholder in the base class.  Concrete subclasses
            provide the actual implementation.
        """
        pass

    def run_iterations(self, iterations_times):
        """Execute a multiple simulation steps from the current node state.
        """
        pass

    def _return_final(self):
        """Post-process raw output into the final result tensor.

        Subclasses override this to convert internal status representations
        into the desired output format.
        """
        pass

class Diffusion_process(MessagePassing):
    """Low-level message-passing engine for opinion dynamics models.

    This class wraps PyTorch Geometric's :class:`~torch_geometric.nn.MessagePassing`
    and manages the edge structure and iteration counter shared by all
    concrete opinion propagation kernels (e.g. ``Voter_process``,
    ``HK_process``).  Subclasses only need to implement :meth:`forward`
    and :meth:`message`.

    :param edge_index: Edge index tensor of shape ``(2, E)``.
    :type edge_index: torch.Tensor
    :param aggr: Aggregation scheme (default ``'sum'``).
    :type aggr: str
    :param kwargs: Model-specific scalars (e.g. ``epsilon``).
        All are stored as instance attributes.
    """

    def __init__(self, edge_index, aggr='sum', **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)
        self.edge_index = edge_index
        self.device = self.edge_index.device
        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _set_iterations(self, iterations_times):
        """Set the number of iterations for the next :meth:`forward` call
        and reset the internal step counter to zero.

        :param iterations_times: Number of time steps to execute.
        :type iterations_times: int
        """
        self.iterations_times = iterations_times
        self.times = 0

