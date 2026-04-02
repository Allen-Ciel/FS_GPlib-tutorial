Opinion Dynamic Models
======================

Opinion dynamics models study how individual opinions evolve through local interactions in a networked population.
Depending on the model, each node holds either a discrete opinion (e.g., binary 0/1) or a continuous opinion
value, and updates it based on the states of its neighbors according to mechanisms such as imitation,
social reinforcement, majority voting, or bounded-confidence averaging.

These models find applications in computational social science, political polarization analysis,
consensus formation, marketing strategy, and the study of echo chambers in online social networks.

All opinion dynamics models in ``FS_GPlib`` operate on **static networks** — directed or undirected graphs
with a fixed topology. The models support execution on both CPU and GPU, and can leverage
the batch-parallel acceleration described in the :doc:`/tutorial`.

Base Class & Common API
-----------------------

Every opinion dynamics model inherits from
:class:`~fs_gplib.Opinions.base.DiffusionModel`,
which provides the shared initialisation pipeline and a uniform simulation
interface.  Understanding the base class helps you leverage the full API of
any concrete model.

.. autoclass:: fs_gplib.Opinions.base.DiffusionModel
   :noindex:

Utility Methods
~~~~~~~~~~~~~~~

The following utility methods can be called on **any** opinion model instance
to re-configure the model after construction:


.. automethod:: fs_gplib.Opinions.base.DiffusionModel._init_node_status
   :noindex:
.. automethod:: fs_gplib.Opinions.base.DiffusionModel._set_device
   :noindex:
.. automethod:: fs_gplib.Opinions.base.DiffusionModel._set_seed
   :noindex:
   
Simulation Interface
~~~~~~~~~~~~~~~~~~~~

All opinion models expose four progressively higher-level simulation methods.
The table below summarises their behaviour; each concrete model page documents
the return type for that specific model.

+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| Method                                    | Description                                               | Resets state?                |
+===========================================+===========================================================+==============================+
| ``Model.run_iteration()``                 | Advance **one** time step from the current state.         | No                           |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_iterations(times)``           | Advance *times* steps from the current state.             | No                           |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_epoch(times)``                | Reset to initial state, then run *times* steps            | Yes                          |
|                                           | (one independent realisation).                            |                              |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+
| ``Model.run_epochs(epochs, times,         | Run *epochs* independent realisations in batches          | Yes                          |
| batch_size)``                             | (Monte-Carlo simulation).                                 |                              |
+-------------------------------------------+-----------------------------------------------------------+------------------------------+

.. tip::

   ``run_iteration`` / ``run_iterations`` do **not** reset node opinions, so
   they can be chained to inspect intermediate states or implement custom
   stopping criteria.  ``run_epoch`` / ``run_epochs`` always start from the
   initial opinion configuration and are the recommended entry points for
   Monte-Carlo studies.

Available Models
----------------

The following models are available:

.. toctree::
   :maxdepth: 1

   models/opinions/Voter.rst
   models/opinions/QVoter.rst
   models/opinions/MajorityRule.rst
   models/opinions/Sznajd.rst
   models/opinions/HK.rst
   models/opinions/WHK.rst




