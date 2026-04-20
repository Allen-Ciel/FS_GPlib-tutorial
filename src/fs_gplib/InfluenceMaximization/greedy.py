"""
Greedy algorithm for Influence Maximization.

The greedy algorithm iteratively selects seed nodes that provide the
maximum marginal increase in expected influence spread. Due to the
submodularity property of influence spread functions, this provides an
approximation guarantee of (1 - 1/e) ≈ 63% of the optimal solution.
"""

import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseInfluenceMaximizer, SpreadEstimator


class GreedyIM(BaseInfluenceMaximizer):
    """Greedy Influence Maximization algorithm.

    Selects seeds iteratively, each time picking the node that maximizes
    the marginal influence spread. Under classical assumptions where the
    influence objective is monotone and submodular, this provides a
    (1 - 1/e) approximation guarantee for the optimal seed set.
    In practice, spread is estimated by Monte-Carlo simulation.

    **Algorithm**:
        S = {}
        for i in range(k):
            for each node v not in S:
                compute marginal_gain = spread(S ∪ {v}) - spread(S)
            S = S ∪ {argmax_v marginal_gain}

    :param model: A diffusion model from ``fs_gplib.Epidemics``.
    :param seed_size: Number of seeds to select.
    :param influenced_type: Set of node state values that count as influenced.
    :param MC: Monte-Carlo simulations per spread estimate.
    :param iterations_times: Diffusion steps per simulation.
    :param verbose: Whether to show progress during selection.

    **Example**::

        from fs_gplib.InfluenceMaximization import GreedyIM
        from fs_gplib.Epidemics import SIRModel

        model = SIRModel(
            data=graph_data,
            seeds=[],
            infection_beta=0.05,
            recovery_lambda=0.01
        )

        im = GreedyIM(
            model=model,
            seed_size=10,
            influenced_type=[1, 2],
            MC=500,
            iterations_times=100
        )
        seeds = im.fit()
    """

    def __init__(
        self,
        model,
        seed_size: int,
        influenced_type: List[int] = None,
        MC: int = 1000,
        iterations_times: int = 100,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            seed_size=seed_size,
            influenced_type=influenced_type,
            MC=MC,
            iterations_times=iterations_times,
            verbose=verbose,
        )

    def fit(self) -> List[int]:
        """Run the greedy algorithm to select seed nodes.

        :return: List of selected seed node indices.
        :rtype: list[int]
        """
        self.selected_seeds = []
        remaining_candidates = set(range(self.num_nodes))

        for i in range(self.seed_size):
            best_node = None
            best_marginal_gain = -np.inf
            current_spread = self._compute_spread(self.selected_seeds, verbose=False)

            candidates = list(remaining_candidates)
            if self.verbose:
                candidates = tqdm(candidates, desc=f"Seed {i+1}/{self.seed_size}")

            for node in candidates:
                trial_seeds = self.selected_seeds + [node]
                new_spread = self._compute_spread(trial_seeds, verbose=False)
                marginal_gain = new_spread - current_spread

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = node

            if best_node is not None:
                self.selected_seeds.append(best_node)
                remaining_candidates.remove(best_node)

                if self.verbose:
                    print(f"  Selected node {best_node}, marginal gain: {best_marginal_gain:.2f}, cumulative: {current_spread + best_marginal_gain:.2f}")

        return self._finalize_fit()

class GreedyIMWithCaching(BaseInfluenceMaximizer):
    """Greedy IM with spread caching to reduce redundant simulations.

    This version caches spread estimates for previously evaluated seed sets,
    significantly reducing computation when the same seed combinations
    are evaluated multiple times.

    :param model: A diffusion model from ``fs_gplib.Epidemics``.
    :param seed_size: Number of seeds to select.
    :param influenced_type: Set of node state values that count as influenced.
    :param MC: Monte-Carlo simulations per spread estimate.
    :param iterations_times: Diffusion steps per simulation.
    :param verbose: Whether to show progress.

    **Example**::

        im = GreedyIMWithCaching(
            model=model,
            seed_size=10,
            influenced_type=[1, 2],
            MC=500
        )
        seeds = im.fit()
        estimator = im.get_estimator()
        print(f"Total spread estimates: {estimator.get_call_count()}")
    """

    def __init__(
        self,
        model,
        seed_size: int,
        influenced_type: List[int] = None,
        MC: int = 1000,
        iterations_times: int = 100,
        verbose: bool = True,
    ):
        super().__init__(
            model=model,
            seed_size=seed_size,
            influenced_type=influenced_type,
            MC=MC,
            iterations_times=iterations_times,
            verbose=verbose,
        )
        self.estimator = SpreadEstimator(self)

    def fit(self) -> List[int]:
        """Run greedy with caching to select seed nodes.

        :return: List of selected seed node indices.
        :rtype: list[int]
        """
        self.selected_seeds = []
        remaining_candidates = set(range(self.num_nodes))
        seeds_tuple = tuple()

        for i in range(self.seed_size):
            best_node = None
            best_marginal_gain = -np.inf
            current_spread = self.estimator.estimate(seeds_tuple)

            candidates = list(remaining_candidates)
            if self.verbose:
                candidates = tqdm(candidates, desc=f"Seed {i+1}/{self.seed_size}")

            for node in candidates:
                trial_tuple = tuple(sorted(seeds_tuple + (node,)))
                new_spread = self.estimator.estimate(trial_tuple)
                marginal_gain = new_spread - current_spread

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_node = node

            if best_node is not None:
                self.selected_seeds.append(best_node)
                seeds_tuple = tuple(sorted(self.selected_seeds))
                remaining_candidates.remove(best_node)

                if self.verbose:
                    print(f"  Selected node {best_node}, marginal gain: {best_marginal_gain:.2f}, cumulative: {current_spread + best_marginal_gain:.2f}")

        return self._finalize_fit()

    def get_estimator(self) -> SpreadEstimator:
        """Return the spread estimator for accessing cache statistics.

        :return: The SpreadEstimator instance.
        :rtype: SpreadEstimator
        """
        return self.estimator

