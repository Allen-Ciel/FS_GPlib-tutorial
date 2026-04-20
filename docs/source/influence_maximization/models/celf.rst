CELF
====

CELF (Cost-Effective Lazy Forward) is a greedy acceleration strategy based on
submodularity. Instead of recomputing every candidate's marginal gain at every
round, CELF maintains a priority queue and performs **lazy re-evaluation** only
when necessary.

Compared with naive greedy selection, CELF-style methods usually reduce the
number of Monte-Carlo spread estimations by a large margin, which directly
improves IM runtime.

Available Algorithms
--------------------

``CELFIM``
~~~~~~~~~~

.. autoclass:: fs_gplib.InfluenceMaximization.celf.CELFIM
   :members:
   :show-inheritance:

``CELFPlusPlus``
~~~~~~~~~~~~~~~~

``CELFPlusPlus`` implements the Greedy CELF++ lazy-forward state machine.
Each candidate keeps four cached fields:

- ``mg1``: marginal gain :math:`\Delta_u(S)` with respect to current seed set.
- ``mg2``: look-ahead marginal gain :math:`\Delta_u(S \cup \{prev\_best\})`.
- ``prev_best``: the best examined node used to compute ``mg2``.
- ``flag``: the iteration in which the cached gains were last refreshed.

In each iteration, ``last_seed`` records the most recently selected seed and
``cur_best`` tracks the best marginal-gain node among candidates examined in
the current iteration. If ``prev_best == last_seed``, ``mg2`` can be reused
directly without recomputing spread.

.. autoclass:: fs_gplib.InfluenceMaximization.celf.CELFPlusPlus
   :members:
   :show-inheritance:

Advanced Interface
------------------

Both ``CELFIM`` and ``CELFPlusPlus`` expose ``get_estimator()`` for
accessing spread-estimation statistics and cache usage.
