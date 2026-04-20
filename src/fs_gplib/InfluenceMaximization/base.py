"""
Base class for Influence Maximization algorithms.

This module provides the foundational infrastructure for seed node selection
using diffusion models from fs_gplib.Epidemics.
"""

import torch
import numpy as np
from typing import Iterable, List, Set
from tqdm import tqdm
import warnings

from ..Epidemics.base import DiffusionModel
from ..utils import check_int


class BaseInfluenceMaximizer:
    """Base class for influence maximization algorithms.

    This class provides the shared infrastructure for computing influence spread
    and managing the seed selection process. Concrete algorithms (Greedy, CELF, etc.)
    inherit from this base class.

    **Core Concept**: The spread range of a seed set :math:`S` is defined as the expected
    number of nodes that reach states specified by ``influenced_type`` when
    diffusion starts from :math:`S`.

    :param model: A diffusion model instance from ``fs_gplib.Epidemics``
        (e.g. ``SIRModel``, ``SIModel``). The model's seeds will be set
        to empty during IM computation.
    :type model: DiffusionModel
    :param seed_size: Number of seed nodes to select.
    :type seed_size: int
    :param influenced_type: Set of node state values that count as "influenced".
        For SIR: ``[1, 2]`` means infected or recovered nodes.
        For SI: ``[1]`` means infected nodes.
    :type influenced_type: list[int] or set[int]
    :param MC: Number of Monte-Carlo simulations for spread estimation.
        Higher values give more accurate estimates but increase computation time.
    :type MC: int
    :param iterations_times: Number of diffusion steps per simulation.
    :type iterations_times: int
    :param verbose: Whether to show progress during computation.
    :type verbose: bool

    .. note::
        The model should be initialized with ``seeds=None``.
        The IM algorithm will set seeds during computation.

    .. warning::
        This implementation is **not thread-safe** when sharing the same
        diffusion model instance across multiple threads/tasks. Influence
        estimation mutates model-internal seed/state via ``model._set_seed(...)``
        before running simulations.

    **Example**::

        from fs_gplib.InfluenceMaximization import GreedyIM
        from fs_gplib.Epidemics import SIRModel

        # Create model with empty seeds
        model = SIRModel(
            data=graph_data,
            seeds=[],
            infection_beta=0.05,
            recovery_lambda=0.01
        )

        # Create IM algorithm
        im = GreedyIM(
            model=model,
            seed_size=10,
            influenced_type=[1, 2],  # Infected + Recovered
            MC=500,
            iterations_times=100
        )

        seeds = im.fit()
    """

    _MODEL_STATE_SPACE = {
        "SIModel": {0, 1},
        "SISModel": {0, 1},
        "ThresholdModel": {0, 1},
        "IndependentCascadesModel": {0, 1},
        "SIRModel": {0, 1, 2},
        "SEISModel": {0, 1, 2},
        "SEISctModel": {0, 1, 2},
        "SEIRModel": {0, 1, 2, 3},
        "SEIRctModel": {0, 1, 2, 3},
        "SWIRModel": {0, 1, 2, 3},
        "ProfileModel": {-1, 0, 1},
        "ProfileThresholdModel": {-1, 0, 1},
        "KerteszThresholdModel": {-1, 0, 1},
    }

    def __init__(
        self,
        model: DiffusionModel,
        seed_size: int,
        influenced_type: List[int] = [1],
        MC: int = 1000,
        iterations_times: int = 100,
        verbose: bool = True,
    ):
        """Initialize the shared IM runtime configuration.

        :param model: Diffusion model used to evaluate spread.
        :type model: DiffusionModel
        :param seed_size: Number of seeds to select.
        :type seed_size: int
        :param influenced_type: Node states treated as influenced.
        :type influenced_type: list[int]
        :param MC: Number of Monte-Carlo runs per spread estimation.
        :type MC: int
        :param iterations_times: Simulation steps in each Monte-Carlo run.
        :type iterations_times: int
        :param verbose: Whether to print progress information.
        :type verbose: bool
        """
        check_int(seed_size=seed_size, MC=MC, iterations_times=iterations_times)

        self._validate_model(model)

        if seed_size <= 0:
            raise ValueError("seed_size must be a positive integer")
        if seed_size > model.data.num_nodes:
            raise ValueError("seed_size must not exceed the number of nodes in the graph")


        self.model = model
        self.seed_size = seed_size
        self.influenced_type = self._validate_influenced_type(model, influenced_type)
        self.MC = MC
        self.iterations_times = iterations_times
        self.device = getattr(model, "device", None)
        self.verbose = verbose

        self.num_nodes = model.data.num_nodes
        self.selected_seeds = []

    def _validate_model(self, model: DiffusionModel):
        """Validate that the model is from fs_gplib.Epidemics module.

        :param model: The diffusion model to validate.
        :type model: DiffusionModel
        :raises TypeError: If model is not a DiffusionModel instance.
        :raises ValueError: If model seeds are not empty.
        """
        from ..Epidemics.base import DiffusionModel as BaseDiffusionModel

        if not isinstance(model, BaseDiffusionModel):
            raise TypeError(
                f"model must be a DiffusionModel from fs_gplib.Epidemics, "
                f"got {type(model).__module__}.{type(model).__name__}"
            )

        if model.seeds is not None:
            model.seeds = None
            warnings.warn(
                "model seeds should be None for IM algorithms. "
                "The IM algorithm will set seeds during computation."
            )


    def _validate_influenced_type(
        self,
        model: DiffusionModel,
        influenced_type: Iterable[int],
    ) -> Set[int]:
        """Validate influenced states and normalize them into a set."""
        if not isinstance(influenced_type, (list, tuple, set)):
            raise TypeError(
                "influenced_type must be a list, tuple, or set of integers"
            )

        if len(influenced_type) == 0:
            raise ValueError("influenced_type cannot be empty")

        normalized = set()
        for state in influenced_type:
            if isinstance(state, bool) or not isinstance(state, int):
                raise TypeError(
                    "influenced_type must contain integers, "
                    f"got {type(state).__name__}"
                )
            normalized.add(state)

        allowed_states = self._MODEL_STATE_SPACE.get(type(model).__name__)
        if allowed_states is not None:
            invalid_states = normalized - allowed_states
            if invalid_states:
                raise ValueError(
                    f"Invalid influenced_type {sorted(invalid_states)} for "
                    f"{type(model).__name__}. Allowed states: "
                    f"{sorted(allowed_states)}"
                )

        return normalized

    def _compute_spread(self, seeds: List[int], verbose: bool = None) -> float:
        """Estimate the expected influence spread of a seed set.

        Runs MC independent simulations and computes the average number of
        nodes that reach states in ``influenced_type`` by the final step.

        :param seeds: List of seed node indices.
        :type seeds: list[int]
        :param verbose: Whether to show a progress bar. Defaults to self.verbose.
        :type verbose: bool
        :return: Average spread across MC simulations.
        :rtype: float

        .. warning::
            This method mutates ``self.model`` internal state by calling
            ``self.model._set_seed(seeds)``. Therefore it is not thread-safe
            to run concurrent spread estimations with the same model instance.
        """
        if verbose is None:
            verbose = self.verbose

        if len(seeds) == 0:
            return 0.0

        # NOTE: non-thread-safe by design.
        # We intentionally mutate shared model state for each trial seed set.
        # Do not call this concurrently with the same model instance.
        self.model._set_seed(seeds)

        batch_size = min(self.MC, 200)
        epoch_groups = [batch_size] * (self.MC // batch_size)
        remainder = self.MC % batch_size
        if remainder > 0:
            epoch_groups.append(remainder)

        total_spread = 0.0
        iterator = tqdm(epoch_groups, desc="Estimating spread") if verbose else epoch_groups

        with torch.no_grad():
            for group_size in iterator:
                finals = self.model.run_epochs(group_size, self.iterations_times, group_size)
                spread = self._count_spread(finals)
                total_spread += spread * group_size

        avg_spread = total_spread / self.MC
        return avg_spread

    def _count_spread(self, finals: torch.Tensor) -> float:
        """Count the average spread from final state tensor.

        Counts nodes that have states in ``influenced_type``.

        :param finals: Tensor of shape ``(MC, N)`` containing final node states.
        :type finals: torch.Tensor
        :return: Average number of nodes in influenced states.
        :rtype: float
        """
        influenced_mask = torch.zeros_like(finals, dtype=torch.bool)
        for state in self.influenced_type:
            influenced_mask = influenced_mask | (finals == state)
        return influenced_mask.float().sum(dim=1).mean().item()

    def fit(self) -> List[int]:
        """Select seed nodes using the configured algorithm.

        This is the main entry point. Subclasses implement the actual
        seed selection logic in this method.

        :return: List of selected seed node indices.
        :rtype: list[int]
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def get_seeds(self) -> List[int]:
        """Return the selected seed nodes.

        :return: List of seed node indices.
        :rtype: list[int]
        """
        return self.selected_seeds.copy()

    def _finalize_fit(self) -> List[int]:
        """Synchronize model seeds and return a defensive copy.

        Called at the end of each ``fit`` implementation so the wrapped
        diffusion model and algorithm output remain consistent.
        """
        if self.selected_seeds:
            self.model._set_seed(self.selected_seeds)
        return self.selected_seeds.copy()


class SpreadEstimator:
    """Utility class for estimating influence spread with caching.

    This class wraps a BaseInfluenceMaximizer and provides memoization
    of spread estimates to avoid redundant Monte-Carlo simulations.

    :param maximizer: The influence maximizer instance.
    :type maximizer: BaseInfluenceMaximizer
    """

    def __init__(self, maximizer: BaseInfluenceMaximizer):
        """Create a spread estimator bound to one maximizer instance.

        :param maximizer: Influence maximizer that performs MC spread calls.
        :type maximizer: BaseInfluenceMaximizer
        """
        self.maximizer = maximizer
        self._cache = {}
        self._call_count = 0

    def estimate(self, seeds: tuple, use_cache: bool = True) -> float:
        """Estimate spread for a seed set, optionally using cached results.

        :param seeds: Sorted tuple of seed node indices.
        :type seeds: tuple[int]
        :param use_cache: Whether to use cached results if available.
        :type use_cache: bool
        :return: Estimated spread.
        :rtype: float
        """
        if not use_cache:
            self._call_count += 1
            return self.maximizer._compute_spread(list(seeds), verbose=False)

        if seeds in self._cache:
            return self._cache[seeds]

        self._call_count += 1
        spread = self.maximizer._compute_spread(list(seeds), verbose=False)
        self._cache[seeds] = spread
        return spread

    def get_cache_size(self) -> int:
        """Return the number of cached spread estimates."""
        return len(self._cache)

    def get_call_count(self) -> int:
        """Return the total number of spread estimates made."""
        return self._call_count

    def clear_cache(self):
        """Clear the spread estimate cache."""
        self._cache.clear()
        self._call_count = 0
