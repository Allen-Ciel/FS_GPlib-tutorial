Dynamic Network Models
======================

In many real-world systems — such as face-to-face contact networks, mobile communication networks,
and transportation systems — the connections between individuals are not fixed but evolve over time.
Dynamic network models capture this by operating on a sequence of graph snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where the node set :math:`V` remains constant while the
edge set :math:`E^{(k)}` may change at each discrete time step :math:`k`.

Each dynamic model in ``FS_GPlib`` extends a corresponding static model from the
:doc:`epidemic` category to time-varying topologies. The propagation rules at each step are
identical to their static counterparts, but the neighbor set and (optionally) edge weights
are drawn from the current snapshot rather than a fixed graph.
This enables the study of how **temporal connectivity patterns** affect spreading outcomes.

All dynamic network models support directed and undirected graphs with optional time-varying edge weights.
They can be executed on both CPU and GPU, and are compatible with the batch-parallel acceleration
described in the :doc:`/tutorial`.

Base Class & Common API
-----------------------

Every dynamic network model inherits from
:class:`~fs_gplib.Dynamic.base.DiffusionModel`,
which provides the shared initialisation pipeline and a uniform simulation
interface.  Understanding the base class helps you leverage the full API of
any concrete model.

Unlike the static-graph base classes, dynamic models take a node tensor *x*
and a list of per-snapshot edge indices *edge_index_list* (instead of a
single PyG ``Data`` object).  The maximum number of simulation steps is
bounded by the number of snapshots :math:`T =` ``len(edge_index_list)``.

.. autoclass:: fs_gplib.Dynamic.base.DiffusionModel
   :noindex:


Utility Methods
~~~~~~~~~~~~~~~

The following utility methods can be called on **any** dynamic model instance
to re-configure the model after construction:

.. automethod:: fs_gplib.Dynamic.base.DiffusionModel._set_seed
   :noindex:
.. automethod:: fs_gplib.Dynamic.base.DiffusionModel._init_node_status
   :noindex:
.. automethod:: fs_gplib.Dynamic.base.DiffusionModel._set_device
   :noindex:

Simulation Interface
~~~~~~~~~~~~~~~~~~~~

All dynamic models expose four progressively higher-level simulation methods.
Note that ``run_epoch`` / ``run_epochs`` always iterate through the **entire
snapshot sequence** and therefore do not accept an ``iterations_times``
parameter (unlike the static-graph models).

+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| Method                                    | Description                                               | Resets state?                |
+===========================================+===========================================================+==============================+
| ``Model.run_iteration()``                 | Advance **one** snapshot step from the current state.     | No                           |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_iterations(times)``           | Advance *times* snapshot steps from the current state.    | No                           |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_epoch()``                     | Reset to initial state, then run through **all**          | Yes                          |
|                                           | snapshots (one independent realisation).                  |                              |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_epochs(epochs,                | Run *epochs* independent realisations in batches,         | Yes                          |
| batch_size)``                             | each covering all snapshots (Monte-Carlo simulation).     |                              |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+

.. tip::

   ``run_iteration`` / ``run_iterations`` do **not** reset node states or the
   snapshot counter, so they can be used to inspect intermediate states or
   implement early-stopping logic.  ``run_epoch`` / ``run_epochs`` always
   start from snapshot 0 with the initial seed configuration and are the
   recommended entry points for Monte-Carlo studies.

Available Models
----------------

The following models are available:

.. toctree::
   :maxdepth: 1

   models/dynamics/DySI.rst
   models/dynamics/DySIS.rst
   models/dynamics/DySIR.rst
   models/dynamics/DySEIR.rst
   models/dynamics/DySEIR_ct.rst
   models/dynamics/DySEIS.rst
   models/dynamics/DySEIS_ct.rst
   models/dynamics/DySWIR.rst
   models/dynamics/DyThreshold.rst
   models/dynamics/DyKThreshold.rst


