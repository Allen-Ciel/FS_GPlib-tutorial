import sys
import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
import warnings

from ..utils import *

class DiffusionModel:
    """Base class for all epidemic / diffusion models on static graphs.

    ``DiffusionModel`` provides the shared initialisation logic and a uniform
    simulation interface that every concrete model (SI, SIR, SEIR, …) inherits.
    Users typically do **not** instantiate this class directly; instead they
    create a subclass such as
    :class:`~fs_gplib.Epidemics.SIModel.SIModel`.

    **Initialisation pipeline** (executed in ``__init__``):

    1. Validate the input graph via :meth:`_validate_graph`.
    2. Initialise seed nodes via :meth:`_initialize_seeds`.
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
        Must contain ``edge_index``; if ``use_weight=True``,
        ``edge_attr`` must also be present.
    :type data: torch_geometric.data.Data
    :param seeds: Initial infected (or activated) node set.
        A **list of node IDs** or a **float in (0, 1)** representing the
        fraction of nodes to infect uniformly at random.
        ``None`` is accepted for models that do not require seeds.
    :type seeds: float | list[int] | None
    :param rand_seed: Random seed for NumPy, used when *seeds* is a
        float to make the random selection reproducible.
        Defaults to ``None``.
    :type rand_seed: int | None
    :param device: ``'cpu'`` or a GPU device index (e.g. ``0``).
        When a GPU index is given the model automatically selects CUDA
        or MPS depending on hardware availability.
        Defaults to ``'cpu'``.
    :type device: str | int
    :param use_weight: If ``True``, edge weights from ``data.edge_attr``
        are used to modulate transmission probabilities.
        Defaults to ``False``.
    :type use_weight: bool
    :param kwargs: Model-specific parameters (e.g. ``infection_beta``,
        ``recovery_lambda``).  Each key-value pair is validated to lie in
        [0, 1] and stored as an instance attribute.
    """

    def __init__(self,
                 data,
                 seeds: Union[float, List[int], None],
                 rand_seed=None,
                 device='cpu',
                 use_weight=False,
                 **kwargs):

        np.random.seed(rand_seed)

        if use_weight not in [True, False]:
            raise ValueError("Parameter 'use_weight' must be either True or False.")
        self.use_weight = use_weight

        self._validate_graph(data)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)

    def _validate_graph(self, data):
        """Validate the input graph and ensure it has the required attributes.

        Checks that *data* is a PyG ``Data`` object with ``edge_index``.
        When ``use_weight`` is enabled, ``edge_attr`` must also exist.
        If ``data.x`` is missing, a zero-filled feature matrix is created
        automatically so that downstream message-passing layers work correctly.

        :param data: The input graph.
        :type data: torch_geometric.data.Data
        :raises ValueError: If *data* is not a ``Data`` object, lacks
            ``edge_index``, or is missing ``edge_attr`` when weights
            are required.
        """
        if not isinstance(data, Data):
            raise ValueError("data must be a Data object from the PyG library.")
        if data.edge_index == None:
            raise ValueError("data must contain edge_index.")
        if self.use_weight:
            if data.edge_attr == None:
                raise ValueError("data does not have edge weights.")
        if data.x == None:
            if data.num_nodes == None:
                num = data.edge_index.max().item()+1
                data.x = torch.zeros((num,1), dtype=torch.long)
            elif isinstance(data.num_nodes, int):
                data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)

        self.data = data

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

        :param seeds: Nodes whose initial state is *Infected* (or *Activated*).

            * **float in (0, 1)** – interpreted as the fraction of nodes to
              select uniformly at random.
            * **list[int]** – explicit node indices.
            * **None** – no seed nodes (used by models that set their own
              initial state).
        :type seeds: float | list[int] | None
        :raises ValueError: If *seeds* is a float outside (0, 1), or the list
            contains non-integer elements.
        """

        self.num_nodes = self._get_num_nodes(self.data)

        if isinstance(seeds, (float, int)):
            if not (0 < seeds < 1):
                raise ValueError("When seeds are decimal numbers, they must be in the range (0,1).")
            seed_count = int(self.num_nodes * seeds)
            random_seeds_list = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
            self.seeds = random_seeds_list#torch.tensor(random_seeds_list)
        elif isinstance(seeds, list):
            if not all(isinstance(i, int) for i in seeds):
                raise ValueError("When seeds are a list, the elements must be integers.")
            if max(seeds) > self.num_nodes:
                seeds = torch.tensor(seeds)
                valid_seeds = seeds[seeds<self.num_nodes]
                removed = seeds.shape[1] - valid_seeds.shape[1]
                if removed > 0:
                    warnings.warn(f"Removed {removed} out_of_range seed index. Valid seed indices are 0 to {self.num_nodes-1}.", UserWarning)
                seeds = valid_seeds.tolist()

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

        :param device: ``'cpu'`` or a GPU device index (e.g. ``0``).
            If a GPU index is given, CUDA is preferred; on Apple Silicon
            the MPS backend is used as a fallback.
        :type device: str | int
        :raises Exception: If a GPU is requested but neither CUDA nor MPS
            is available.

        .. rubric:: Example

        .. code-block:: python

            model = SIModel(data, seeds, infection_beta=0.05, device='cpu')
            # switch to GPU 0
            model._set_device(0)
        """
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:{}'.format(device))
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps:{}'.format(device))
            else:
                raise Exception("No supported GPU device (MPS or CUDA) is available.")
    def _validate_parameters(self, kwargs):
        """Validate and store model-specific parameters.

        Every keyword argument is checked to be a float in [0, 1] and then
        stored as an instance attribute with the same name.

        :param kwargs: Model-specific parameters (e.g.
            ``infection_beta=0.01``).
        :type kwargs: dict
        :raises SystemExit: If any value is outside [0, 1].
        """
        try:
            check_float_parameter(0, 1, True, True, **kwargs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _init_node_status(self):
        """Reset all node states to the initial configuration.

        This method rebuilds the internal ``node_status`` tensor(s) so that
        only the seed nodes are in the *Infected* (or *Activated*) state and
        all other nodes are *Susceptible*.

        Subclasses **must** override this method to create the status
        tensors appropriate for their compartmental structure (e.g. a single
        boolean tensor for SI, or a dict of boolean tensors for SIR).

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
        """Replace the current seed set and refresh node states.

        This is the recommended way to change seed nodes **after**
        construction.  It calls :meth:`_initialize_seeds` to parse and store
        the new seeds, then (if the model is fully initialised)
        :meth:`_init_node_status` to rebuild the node-state tensors on the
        current device.

        :param seeds: New seed specification – same format as the constructor
            parameter *seeds* (a float fraction, a list of node IDs, or
            ``None``).
        :type seeds: float | list[int] | None

        .. rubric:: Example

        .. code-block:: python

            model = SIRModel(data, seeds=[0, 1, 2], ...)
            # later, switch to a random 10 % seed set
            model._set_seed(0.1)
        """
        self._initialize_seeds(seeds)
        if hasattr(self, 'device'):
            self._init_node_status()

    def run_epoch(self, **kwargs):
        """Run a single Monte-Carlo epoch (one independent realisation).

        Node states are **re-initialised** before the epoch starts, so
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
        try:
            check_int(iterations_times=iterations_times, epochs=epochs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

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


    def run_iteration(self):
        """Execute a single simulation step from the current node state.

        The internal ``node_status`` is updated in-place so that
        subsequent calls continue from the latest state.

        .. note::
            This is a placeholder in the base class.  Concrete subclasses
            provide the actual implementation.
        """
        pass

    def _return_final(self):
        """Post-process raw output into the final result tensor.

        Subclasses override this to convert internal status representations
        (e.g. multiple boolean masks) into a single integer state tensor.
        """
        pass

class Diffusion_process(MessagePassing):
    """Low-level message-passing engine for diffusion models.

    This class wraps PyTorch Geometric's :class:`~torch_geometric.nn.MessagePassing`
    and manages the edge structure, edge weights, and iteration counter
    shared by all concrete propagation kernels (e.g. ``SI_process``,
    ``SIR_process``).  Subclasses only need to implement :meth:`forward`
    and :meth:`message`.

    :param edge_index: Edge index tensor of shape ``(2, E)``.
    :type edge_index: torch.Tensor
    :param aggr: Aggregation scheme (default ``'add'``).
    :type aggr: str
    :param kwargs: Model-specific scalars (e.g. ``infection_beta``) and
        optionally ``edge_attr``.  All are stored as instance attributes.
    """

    def __init__(self, edge_index, aggr='add', **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)
        self.edge_attr = None
        self.edge_index = edge_index
        self.device = self.edge_index.device
        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.unsqueeze(0).unsqueeze(2)
        else:
            self.edge_attr = torch.tensor([1]).to(
                self.device)

    def _set_iterations(self, iterations_times):
        """Set the number of iterations for the next :meth:`forward` call
        and reset the internal step counter to zero.

        :param iterations_times: Number of time steps to execute.
        :type iterations_times: int
        """
        self.iterations_times = iterations_times
        self.times = 0