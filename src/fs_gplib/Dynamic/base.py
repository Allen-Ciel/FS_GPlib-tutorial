import sys
import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from tqdm import tqdm


from ..utils import *

class DiffusionModel:
    """Base class for all dynamic-network diffusion models.

    ``DiffusionModel`` (in the Dynamic package) provides the shared
    initialisation logic and a uniform simulation interface that every
    concrete dynamic model (DySI, DySIR, DySEIR, DyThreshold, …) inherits.
    Users typically do **not** instantiate this class directly; instead they
    create a subclass such as
    :class:`~fs_gplib.Dynamic.DySIRModel.DySIRModel`.

    Unlike the static-graph base classes in the Epidemics and Opinions
    packages, dynamic models operate on a **snapshot sequence**
    :math:`\\{G^{(k)}=(V,E^{(k)})\\}_{k=1}^{T}` where the edge set changes
    at every step while the node set :math:`V` remains fixed.  The input
    is therefore a node tensor *x* and a list of per-snapshot edge indices
    *edge_index_list* (with optional *edge_attr_list* for time-varying
    weights).

    **Initialisation pipeline** (executed in ``__init__``):

    1. Validate the input graph data via :meth:`_validate_graph`.
    2. Initialise seed nodes via :meth:`_initialize_seeds`.
    3. Validate and store model-specific parameters via
       :meth:`_validate_parameters`.
    4. Transfer the model to the target device via :meth:`_set_device`.

    **Simulation interface** – four progressively higher-level entry points:

    +-----------------------------------------+-----------------------------------------------------------+
    | Method                                  | Description                                               |
    +=========================================+===========================================================+
    | :meth:`run_iteration`                   | Advance **one** snapshot step from the current state.     |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_iterations(times)            | Advance *times* snapshot steps from the current state.    |
    | <run_iteration>`                        |                                                           |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_epoch()                      | Reset state, then run through **all** snapshots           |
    | <run_epoch>`                            | (one independent realisation).                            |
    +-----------------------------------------+-----------------------------------------------------------+
    | :meth:`run_epochs(epochs,               | Run *epochs* independent realisations in batches          |
    | batch_size) <run_epochs>`               | (Monte-Carlo simulation over all snapshots).              |
    +-----------------------------------------+-----------------------------------------------------------+

    :param x: Node feature tensor of shape ``(N, 1)``.  The leading
        dimension defines the node count :math:`N`.
    :type x: torch.Tensor
    :param edge_index_list: A list of ``edge_index`` tensors, one per
        snapshot.  The list length :math:`T` determines the maximum number
        of simulation steps.
    :type edge_index_list: list[torch.Tensor]
    :param seeds: Initial infected (or activated) node set.
        A **list of node IDs** or a **float in [0, 1)** representing the
        fraction of nodes to infect uniformly at random.
        ``None`` is accepted for models that do not require seeds.
    :type seeds: float | list[int] | None
    :param rand_seed: Random seed for NumPy, used when *seeds* is a
        float to make the random selection reproducible.
        Defaults to ``None``.
    :type rand_seed: int | None
    :param device: ``'cpu'`` or a CUDA device index (e.g. ``0``).
        Defaults to ``'cpu'``.
    :type device: str | int
    :param edge_attr_list: *(optional)* A list of per-snapshot edge-weight
        tensors aligned with *edge_index_list*.  If ``None``, all edge
        weights default to ``1``.
    :type edge_attr_list: list[torch.Tensor] | None
    :param kwargs: Model-specific parameters (e.g. ``infection_beta``,
        ``recovery_lambda``).  Each key-value pair is validated to lie in
        [0, 1] and stored as an instance attribute.
    """

    def __init__(self,
                 x,
                 edge_index_list,
                 seeds: Union[float, List[int], None],
                 rand_seed=None,
                 device='cpu',
                 edge_attr_list=None,
                 **kwargs):
        np.random.seed(rand_seed)

        self.use_weight = True
        if edge_attr_list is None:
            self.use_weight = False

        self._validate_graph(x, edge_index_list, edge_attr_list)

        self.x = x
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list

        np.random.seed(rand_seed)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)



    def _validate_graph(self, x, edge_index_list, edge_attr_list=None):
        """Validate the input graph data for dynamic-network simulation.

        Checks that *x* is a ``torch.Tensor``, *edge_index_list* is a
        non-empty list of 2-row ``edge_index`` tensors, and (when provided)
        *edge_attr_list* is aligned with *edge_index_list* in both length
        and per-snapshot edge count.

        :param x: Node feature tensor of shape ``(N, ...)``.
        :type x: torch.Tensor
        :param edge_index_list: Per-snapshot edge indices.
        :type edge_index_list: list[torch.Tensor]
        :param edge_attr_list: *(optional)* Per-snapshot edge weights.
        :type edge_attr_list: list[torch.Tensor] | None
        :raises TypeError: If *x* is not a tensor or *edge_index_list*
            is not a list.
        :raises ValueError: If *edge_index_list* is empty, any element
            has an unexpected shape, or *edge_attr_list* is misaligned.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if not isinstance(edge_index_list, list) or len(edge_index_list) == 0:
            raise ValueError("edge_index_list must be a non-empty list of edge_index tensors.")
        for k, ei in enumerate(edge_index_list):
            if not isinstance(ei, torch.Tensor):
                raise TypeError(f"edge_index_list[{k}] must be a torch.Tensor.")
            if ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(f"edge_index_list[{k}] must have shape (2, E), got {tuple(ei.shape)}.")
        if edge_attr_list is not None:
            if not isinstance(edge_attr_list, list):
                raise TypeError("edge_attr_list must be a list of tensors or None.")
            if len(edge_attr_list) != len(edge_index_list):
                raise ValueError(
                    f"edge_attr_list length ({len(edge_attr_list)}) must match "
                    f"edge_index_list length ({len(edge_index_list)})."
                )
            for k, (ei, ea) in enumerate(zip(edge_index_list, edge_attr_list)):
                if ea.shape[0] != ei.shape[1]:
                    raise ValueError(
                        f"edge_attr_list[{k}] length ({ea.shape[0]}) does not match "
                        f"edge_index_list[{k}] edge count ({ei.shape[1]})."
                    )

    def _get_num_nodes(self):
        """Return the number of nodes :math:`N` (inferred from ``self.x``).

        :return: Number of nodes.
        :rtype: int
        """
        return self.x.shape[0]

    def _initialize_seeds(self, seeds):
        """Parse *seeds* and store the result in ``self.seeds``.

        This low-level method only updates ``self.seeds`` without refreshing
        node-state tensors.  For post-construction use, prefer
        :meth:`_set_seed` which also calls :meth:`_init_node_status`
        automatically.

        :param seeds: Nodes whose initial state is *Infected* (or *Activated*).

            * **float in [0, 1)** – fraction of nodes selected uniformly at
              random.
            * **list[int]** – explicit node indices.
            * **None** – no seed nodes.
        :type seeds: float | list[int] | None
        :raises ValueError: If *seeds* is a float outside [0, 1), the list
            contains non-integer elements, or indices exceed graph size.
        """

        self.num_nodes = self.x.shape[0]

        if isinstance(seeds, (float, int)):
            if not (0 <= seeds < 1):
                raise ValueError("When seeds are decimal numbers, they must be in the range (0,1).")
            seed_count = int(self.num_nodes * seeds)
            self.seeds = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
        elif isinstance(seeds, list):
            if not all(isinstance(i, int) for i in seeds):
                raise ValueError("When seeds are a list, the elements must be integers.")
            if max(seeds) >= self.num_nodes:
                raise ValueError("The maximum value in the seeds list cannot exceed the number of nodes in data.")
            self.seeds = seeds
        elif seeds is None:
            self.seeds = seeds
        else:
            raise ValueError("seeds must be decimals in the range (0,1), a list of integers, or None.")

    def _set_device(self, device):
        """Set (or switch) the computation device for the model.

        When called during initialisation the internal tensors are moved to
        the target device.  Subclasses override this method to also move
        the message-passing process module to the device.

        :param device: ``'cpu'`` or a CUDA device index (e.g. ``0``).
        :type device: str | int
        :raises Exception: If a GPU is requested but CUDA is not available.

        .. rubric:: Example

        .. code-block:: python

            model = DySIRModel(x, edge_index_list, seeds, ...)
            model._set_device(0)  # switch to GPU 0
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
        tensors appropriate for their compartmental structure.

        Calling this method is useful when you want to **rerun** a simulation
        from scratch without reconstructing the model object.

        .. rubric:: Example

        .. code-block:: python

            result1 = model.run_iterations(5)
            model._init_node_status()   # reset
            result2 = model.run_iterations(5)
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

            model = DySIRModel(x, edge_index_list, seeds=[0, 1], ...)
            # switch to a random 10 % seed set
            model._set_seed(0.1)
        """
        self._initialize_seeds(seeds)
        if hasattr(self, 'device'):
            self._init_node_status()

    def run_epoch(self, **kwargs):
        """Run one Monte-Carlo realisation over the **full** snapshot sequence.

        The process internal step counter is reset; node states are
        **re-initialised** before the epoch starts.

        .. note::
            This is a placeholder in the base class.  Concrete subclasses
            provide the actual implementation.
        """
        pass

    def run_epochs(self, epochs, iterations_times):
        """Run multiple independent Monte-Carlo epochs sequentially.

        This is a simplified version; most concrete models override
        :meth:`run_epochs` with a batch-parallel implementation that runs
        through **all** snapshots for each realisation.

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


    def iteration(self):
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
    """Low-level message-passing engine for dynamic-network models.

    This class wraps PyTorch Geometric's :class:`~torch_geometric.nn.MessagePassing`
    and manages the snapshot sequence, edge weights, and internal step counter
    shared by all concrete dynamic propagation kernels (e.g.
    ``DySIModel_process``, ``DySIRModel_process``).  Subclasses only need to
    implement :meth:`forward` and :meth:`message`.

    :param edge_index_list: List of per-snapshot ``edge_index`` tensors.
    :type edge_index_list: list[torch.Tensor]
    :param edge_attr_list: List of per-snapshot edge-weight tensors, or
        ``None`` if edges are unweighted.
    :type edge_attr_list: list[torch.Tensor] | None
    :param aggr: Aggregation scheme (default ``'add'``).
    :type aggr: str
    :param kwargs: Model-specific scalars (e.g. ``infection_beta``).
        All are stored as instance attributes.
    """

    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 aggr='add',
                 **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)

        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list

        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _set_iterations(self):
        """Reset the internal snapshot step counter to zero.

        Called before each Monte-Carlo realisation so that
        :meth:`forward` starts from the first snapshot.
        """
        self.times = 0