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

    import torch
    from torch_geometric.datasets import Planetoid
    from fs_gplib.Epidemics import SIRModel
    from fs_gplib.InfluenceMaximization import GreedyIM, CELFIM

    # 1) Load Cora graph
    dataset = Planetoid(root="./dataset", name="Cora")
    data = dataset[0]

    # 2) Build diffusion model for Greedy (seeds must be None)
    model = SIRModel(
        data=data,
        seeds=None,
        infection_beta=0.05,
        recovery_lambda=0.01,
        use_weight=False
    )

    # 3) Greedy baseline
    greedy = GreedyIM(
        model=model,
        seed_size=20,
        influenced_type=[1, 2],  # Infected + Recovered
        MC=400,
        iterations_times=100,
        verbose=True,
    )
    greedy_seeds = greedy.fit()

    model._set_seed(None)
    # 4) Faster CELF alternative
    celf = CELFIM(
        model=model,
        seed_size=20,
        influenced_type=[1, 2],
        MC=400,
        iterations_times=100,
        verbose=True,
    )
    celf_seeds = celf.fit()

    print("Greedy seeds:", greedy_seeds[:5], "...")
    print("CELF seeds:", celf_seeds[:5], "...")

Algorithms
----------

.. toctree::
   :maxdepth: 1

   models/greedy
   models/celf
