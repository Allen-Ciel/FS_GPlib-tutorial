Influence Maximization
======================

Overview
--------

Influence Maximization (IM) asks a classical question in network science:
given a budget :math:`k`, which :math:`k` seed nodes should be activated so that
the expected propagation range is maximized under a diffusion model.

This problem is central to viral marketing, public opinion guidance,
information campaign planning, and epidemic intervention. For common diffusion
settings (for example Independent Cascade and threshold families), the spread
objective is monotone and submodular under classical assumptions, enabling a
greedy strategy with :math:`(1 - 1/e)` approximation guarantee in that setting.
In practical implementations, spread is typically estimated by Monte-Carlo
simulation.

In practice, the main runtime cost of greedy-style IM comes from repeatedly
estimating spread through propagation simulation. ``FS_GPlib`` implements these
simulation computations efficiently, so greedy algorithms that rely on repeated
propagation estimation can be significantly accelerated:

- ``GreedyIM``: standard greedy baseline with explicit marginal-gain evaluation.
- ``GreedyIMWithCaching``: avoids repeated spread computation on duplicate seed sets.
- ``CELFIM``: lazy-forward strategy that dramatically reduces re-evaluation.
- ``CELFPlusPlus``: CELF++ lazy-forward variant with cached look-ahead gains.

Compared with naive greedy evaluation, CELF-style methods typically achieve
substantial speedups while preserving nearly identical seed quality in most
benchmark settings.

Base Class & Common API
-----------------------

All IM algorithms inherit from
:class:`~fs_gplib.InfluenceMaximization.base.BaseInfluenceMaximizer`,
which provides a shared workflow for:

- model compatibility checks and influenced-state validation;
- estimating expected spread by Monte-Carlo simulation;
- consistent ``fit()`` / ``get_seeds()`` workflow.


.. autoclass:: fs_gplib.InfluenceMaximization.base.BaseInfluenceMaximizer
   :noindex:

Common Workflow
~~~~~~~~~~~~~~~

All IM algorithms expose the same user-facing workflow:

+-------------------+--------------------------------------------------+
| Method            | Description                                      |
+===================+==================================================+
| ``fit()``         | Run the seed-selection algorithm and return      |
|                   | the selected seed nodes.                         |
+-------------------+--------------------------------------------------+
| ``get_seeds()``   | Return the selected seeds after ``fit()``.       |
+-------------------+--------------------------------------------------+

.. automethod:: fs_gplib.InfluenceMaximization.base.BaseInfluenceMaximizer.fit
   :noindex:

.. automethod:: fs_gplib.InfluenceMaximization.base.BaseInfluenceMaximizer.get_seeds
   :noindex:

.. note::

   IM algorithms mutate the wrapped diffusion model by updating its seed set
   during spread estimation. If you want to compare multiple IM algorithms,
   create a fresh diffusion model instance for each run.

.. note::

   ``influenced_type`` valid states depend on the diffusion model.
   Typical choices include ``[1, 2]`` for ``SIRModel`` and ``[1]`` for ``SIModel``.

Quick Example
-------------

The example below shows a practical IM workflow.

.. code-block:: python

    import networkx as nx
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx

    from src.fs_gplib.Epidemics import SIRModel
    from src.fs_gplib.InfluenceMaximization import CELFIM


    def nx_to_pyg_data_with_mapping(graph: nx.Graph) -> tuple[Data, dict, dict]:
        """Convert a NetworkX graph to PyG data with bidirectional mappings."""
        original_nodes = list(graph.nodes())
        node_mapping = {node_id: idx for idx, node_id in enumerate(original_nodes)}
        reverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}

        relabeled_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
        data = from_networkx(relabeled_graph)
        data.num_nodes = relabeled_graph.number_of_nodes()
        return data, node_mapping, reverse_mapping


    def run_im_with_api(
        data: Data,
        node_mapping: dict,
        reverse_mapping: dict,
        seed_size: int = 5,
    ):
        """Run influence maximization via the fs_gplib API (CELFIM + SIRModel)."""

        model = SIRModel(
            data=data,
            seeds=None,
            infection_beta=0.01,
            recovery_lambda=0.005,
        )

        im = CELFIM(
            model=model,
            seed_size=seed_size,
            influenced_type=[1, 2],  # In SIR, both infected and recovered count as influenced.
            MC=1000,
            iterations_times=50,
            verbose=True,
        )

        selected_seeds = im.fit()
        original_id_seeds = [reverse_mapping[s] for s in selected_seeds]
        print(f"selected seeds (reindexed): {selected_seeds}")
        print(f"selected seeds (original ids): {original_id_seeds}")
        print(f"node mapping size: {len(node_mapping)}")
        return selected_seeds


    if __name__ == "__main__":
        graph = nx.les_miserables_graph()
        data, node_mapping, reverse_mapping = nx_to_pyg_data_with_mapping(graph)
        seeds = run_im_with_api(
            data=data,
            node_mapping=node_mapping,
            reverse_mapping=reverse_mapping,
            seed_size=5,
        )
        print(f"seed count = {len(seeds)}")



Algorithms
----------

.. toctree::
   :maxdepth: 1

   models/greedy
   models/celf
