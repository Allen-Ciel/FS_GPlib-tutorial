"""
CELF (Cost-Effective Lazy Forward) algorithm for Influence Maximization.

CELF exploits the submodularity of the influence spread function to achieve
significant speedup over the naive greedy algorithm through lazy evaluation.
It uses a priority queue (max-heap) and only re-evaluates nodes when their
marginal gain could potentially exceed previously evaluated bounds.
"""

import heapq
import torch
import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseInfluenceMaximizer, SpreadEstimator


class CELFNode:
    """Wrapper for a node in the CELF priority queue.

    Stores the node index, its marginal gain estimate, and the iteration
    at which it was last evaluated. Used for lazy evaluation.

    :param node: Node index.
    :param marginal_gain: Current marginal gain estimate.
    :param last_eval_iter: Iteration of last evaluation.
    """

    __slots__ = ['node', 'marginal_gain', 'last_eval_iter', 'prev_best', 'second_gain']

    def __init__(
        self,
        node: int,
        marginal_gain: float,
        last_eval_iter: int,
        prev_best: int = None,
        second_gain: float = None,
    ):
        self.node = node
        self.marginal_gain = marginal_gain
        self.last_eval_iter = last_eval_iter
        self.prev_best = prev_best
        self.second_gain = second_gain

    def __repr__(self):
        return (
            f"CELFNode(node={self.node}, gain={self.marginal_gain:.2f}, "
            f"iter={self.last_eval_iter}, prev_best={self.prev_best}, "
            f"second_gain={self.second_gain})"
        )

    def __lt__(self, other: 'CELFNode'):
        if self.marginal_gain != other.marginal_gain:
            return self.marginal_gain > other.marginal_gain
        return self.node < other.node


class CELFIM(BaseInfluenceMaximizer):
    """CELF (Cost-Effective Lazy Forward) Influence Maximization.

    CELF dramatically reduces the number of spread evaluations compared to
    the naive greedy algorithm by exploiting two properties:

    1. **Submodularity**: The marginal gain of adding a node decreases as
       more nodes are already selected.

    2. **Lazy Evaluation**: Using a max-heap, CELF only recomputes marginal
       gains when they could potentially exceed the current best. Many nodes
       are "lazy" and never get re-evaluated after their initial estimate.

    This results in O(k * n) average-case complexity instead of O(k * n * MC)
    spread estimates, where k is seed size, n is number of nodes, and MC is
    Monte-Carlo simulations.

    :param model: A diffusion model from ``fs_gplib.Epidemics``.
    :param seed_size: Number of seeds to select.
    :param influenced_type: Set of node state values that count as influenced.
    :param MC: Monte-Carlo simulations per spread estimate.
    :param iterations_times: Diffusion steps per simulation.
    :param verbose: Whether to show progress and statistics.

    **Example**::

        from fs_gplib.InfluenceMaximization import CELFIM
        from fs_gplib.Epidemics import SIRModel

        model = SIRModel(
            data=graph_data,
            seeds=[],
            infection_beta=0.05,
            recovery_lambda=0.01
        )

        im = CELFIM(
            model=model,
            seed_size=10,
            influenced_type=[1, 2],
            MC=500,
            iterations_times=100
        )
        seeds = im.fit()
        print(f"Total spread estimates: {im.estimator.get_call_count()}")
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
        """Run CELF algorithm to select seed nodes.

        Uses a max-heap with lazy evaluation to minimize spread estimates.
        Implements the original CELF logic from Leskovec et al. (KDD 2007):
        each node's ``marginal_gain`` always stores the true marginal gain
        ``f(S ∪ {v}) - f(S)`` relative to the seed set at the time of its
        last evaluation.  A node is re-evaluated only when its cached gain
        is from a previous round (i.e. stale due to submodularity).

        :return: List of selected seed node indices.
        :rtype: list[int]
        """
        self.selected_seeds = []
        seeds_set = set()
        seeds_tuple = tuple()

        if self.verbose:
            print(f"Initializing CELF with {self.num_nodes} candidate nodes...")

        # Initialize heap: with S = ∅, f(∅) = 0, so marginal gain = f({v}).
        # last_eval_iter = 0 marks these as evaluated in round 0 (before any
        # seed is selected), matching current_round = len(selected_seeds) = 0
        # on the first outer-loop iteration so no re-evaluation is triggered.
        heap = []
        for node in range(self.num_nodes):
            marginal_gain = self.estimator.estimate((node,))
            celf_node = CELFNode(node, marginal_gain, 0)
            heap.append(celf_node)
        heapq.heapify(heap)

        while len(self.selected_seeds) < self.seed_size:
            current_round = len(self.selected_seeds)
            current_spread = self.estimator.estimate(seeds_tuple) if seeds_tuple else 0.0

            if self.verbose:
                total_estimates = self.estimator.get_call_count()
                top_node = heap[0]
                print(f"\n--- Seed {current_round + 1}/{self.seed_size} ---")
                print(f"  Top of heap: node {top_node.node}, gain={top_node.marginal_gain:.2f} "
                      f"(estimates so far: {total_estimates})")

            # Pop nodes until we find one whose marginal gain was computed
            # against the *current* seed set (last_eval_iter == current_round).
            # Any stale node is re-evaluated and pushed back; by submodularity
            # its new gain is ≤ the cached value, so the heap ordering is
            # maintained as an upper-bound priority queue.
            while True:
                celf_node = heapq.heappop(heap)

                if celf_node.node in seeds_set:
                    continue

                if celf_node.last_eval_iter < current_round:
                    # Gain is stale — recompute true marginal gain f(S∪{v})-f(S)
                    trial_seeds = tuple(sorted(seeds_tuple + (celf_node.node,)))
                    new_total = self.estimator.estimate(trial_seeds)
                    celf_node.marginal_gain = new_total - current_spread
                    celf_node.last_eval_iter = current_round
                    heapq.heappush(heap, celf_node)
                else:
                    # Gain is fresh for this round — this is the best candidate
                    break

            self.selected_seeds.append(celf_node.node)
            seeds_set.add(celf_node.node)
            seeds_tuple = tuple(sorted(self.selected_seeds))

            if self.verbose:
                total_spread = current_spread + celf_node.marginal_gain
                print(f"  Selected node {celf_node.node}, "
                      f"marginal gain={celf_node.marginal_gain:.2f}, "
                      f"total spread={total_spread:.2f}")

        if self.verbose:
            print(f"\n=== CELF Complete ===")
            print(f"Total spread estimates: {self.estimator.get_call_count()}")
            print(f"Cache size: {self.estimator.get_cache_size()}")

        return self._finalize_fit()

    def get_estimator(self) -> SpreadEstimator:
        """Return the spread estimator for accessing statistics.

        :return: The SpreadEstimator instance.
        :rtype: SpreadEstimator
        """
        return self.estimator

class CELFPlusPlus(BaseInfluenceMaximizer):
    """CELF++ (Cost-Effective Lazy Forward++) for Influence Maximization.

    :param model: A diffusion model from ``fs_gplib.Epidemics``.
    :param seed_size: Number of seeds to select.
    :param influenced_type: Set of node state values that count as influenced.
    :param MC: Monte-Carlo simulations per spread estimate.
    :param iterations_times: Diffusion steps per simulation.
    :param verbose: Whether to show progress.

    **Example**::

        from fs_gplib.InfluenceMaximization import CELFPlusPlus

        im = CELFPlusPlus(
            model=model,
            seed_size=10,
            influenced_type=[1, 2]
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
        self.estimator = SpreadEstimator(self)

    def fit(self) -> List[int]:
        """Run CELF++ algorithm to select seed nodes.

        Implements the Greedy CELF++ state machine from the paper:
        each candidate stores ``mg1`` (as ``marginal_gain``), ``mg2``
        (as ``second_gain``), ``prev_best``, and ``flag`` (as
        ``last_eval_iter``). During lazy updates, stale candidates can
        reuse ``mg2`` when ``prev_best == last_seed``.

        Variables follow the pseudocode semantics:
        ``S`` is the current seed set, ``last_seed`` is the most recently
        selected seed, and ``cur_best`` tracks the node with the maximum
        marginal gain among candidates examined in the current iteration.

        :return: List of selected seed node indices.
        :rtype: list[int]
        """
        self.selected_seeds = []
        seeds_set = set()
        seeds_tuple = tuple()
        last_seed = None
        cur_best = None

        if self.verbose:
            print(f"Initializing CELF++...")

        # Initialize queue entries with mg1/mg2/prev_best/flag (flag=0).
        heap = []
        best_init_gain = -np.inf
        current_spread = 0.0
        for node in range(self.num_nodes):
            mg1 = self.estimator.estimate((node,))
            prev_best = cur_best

            if cur_best is None:
                mg2 = mg1
            else:
                pair_seeds = tuple(sorted((node, cur_best)))
                spread_with_pair = self.estimator.estimate(pair_seeds)
                spread_with_best = self.estimator.estimate((cur_best,))
                mg2 = spread_with_pair - spread_with_best

            celf_node = CELFNode(
                node=node,
                marginal_gain=mg1,
                last_eval_iter=0,
                prev_best=prev_best,
                second_gain=mg2,
            )
            heap.append(celf_node)

            if mg1 > best_init_gain:
                best_init_gain = mg1
                cur_best = node

        heapq.heapify(heap)

        while len(self.selected_seeds) < self.seed_size:
            current_round = len(self.selected_seeds)
            current_spread = self.estimator.estimate(seeds_tuple) if seeds_tuple else 0.0
            # cur_best is iteration-local: best examined candidate for current S.
            cur_best = None
            cur_best_gain = -np.inf

            if self.verbose:
                top_node = heap[0]
                print(f"\n--- Seed {current_round + 1}/{self.seed_size} ---")
                print(f"  Top of heap: node {top_node.node}, gain={top_node.marginal_gain:.2f} "
                      f"(estimates so far: {self.estimator.get_call_count()})")

            while True:
                celf_node = heapq.heappop(heap)

                if celf_node.node in seeds_set:
                    continue

                if celf_node.last_eval_iter == current_round:
                    # Fresh for this round: accept as next seed.
                    break

                if celf_node.prev_best == last_seed:
                    # Use cached mg2 without re-estimating.
                    celf_node.marginal_gain = celf_node.second_gain
                else:
                    # mg1 = Δ_u(S) = f(S∪{u}) - f(S)
                    trial_seeds = tuple(sorted(seeds_tuple + (celf_node.node,)))
                    spread_with_u = self.estimator.estimate(trial_seeds)
                    celf_node.marginal_gain = spread_with_u - current_spread

                    # mg2 = Δ_u(S∪{cur_best}) when cur_best is available.
                    celf_node.prev_best = cur_best
                    if cur_best is None or cur_best == celf_node.node or cur_best in seeds_set:
                        celf_node.second_gain = celf_node.marginal_gain
                    else:
                        seeds_with_best = tuple(sorted(seeds_tuple + (cur_best,)))
                        spread_with_best = self.estimator.estimate(seeds_with_best)
                        seeds_with_best_and_u = tuple(sorted(seeds_tuple + (cur_best, celf_node.node)))
                        spread_with_best_and_u = self.estimator.estimate(seeds_with_best_and_u)
                        celf_node.second_gain = spread_with_best_and_u - spread_with_best

                celf_node.last_eval_iter = current_round
                if celf_node.marginal_gain > cur_best_gain:
                    cur_best_gain = celf_node.marginal_gain
                    cur_best = celf_node.node
                heapq.heappush(heap, celf_node)

            self.selected_seeds.append(celf_node.node)
            seeds_set.add(celf_node.node)
            seeds_tuple = tuple(sorted(self.selected_seeds))
            last_seed = celf_node.node

            if self.verbose:
                total_spread = self.estimator.estimate(seeds_tuple)
                print(f"  Seed {len(self.selected_seeds)}: node {celf_node.node}, "
                      f"marginal gain={celf_node.marginal_gain:.2f}, "
                      f"total spread={total_spread:.2f}")

        if self.verbose:
            print(f"\nTotal spread estimates: {self.estimator.get_call_count()}")

        return self._finalize_fit()

    def get_estimator(self) -> SpreadEstimator:
        """Return the spread estimator for accessing statistics."""
        return self.estimator

