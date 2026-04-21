Greedy
======

Greedy-family methods iteratively construct the seed set by maximizing
the marginal spread gain in each round. They directly optimize:

.. math::

   \Delta(v \mid S) = f(S \cup \{v\}) - f(S)

where :math:`f(\cdot)` is the expected influence spread under the selected
diffusion model.


Under classical assumptions where the influence objective is monotone and
submodular, the standard greedy algorithm provides a
:math:`(1 - 1/e)` approximation guarantee [1]_. In engineering practice, the
objective value is estimated by Monte-Carlo simulation. ``FS_GPlib`` provides
both the vanilla version and a caching-enhanced variant.

Available Algorithms
--------------------

``GreedyIM``
~~~~~~~~~~~~

.. autoclass:: fs_gplib.InfluenceMaximization.greedy.GreedyIM
   :members:
   :show-inheritance:

``GreedyIMWithCaching``
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fs_gplib.InfluenceMaximization.greedy.GreedyIMWithCaching
   :members:
   :show-inheritance:

Advanced Interface
------------------

``GreedyIMWithCaching`` exposes ``get_estimator()`` for inspecting cache
statistics (for example total spread-estimation calls and cache size).

References
----------

.. [1] Kempe D, Kleinberg J, Tardos É. Maximizing the spread of influence through a social network[C]//Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining. 2003: 137-146.