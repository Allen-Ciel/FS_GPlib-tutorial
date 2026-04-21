Epidemic Models
===============

Epidemic models originate from mathematical epidemiology and describe how diseases, information,
or behaviors spread through a population represented as a network.
In these models, each node occupies a discrete state (e.g., Susceptible, Infected, Recovered)
and transitions between states according to stochastic rules governed by local interactions with neighbors.

Epidemic models have broad applications beyond public health, including viral marketing,
rumor propagation, cybersecurity threat analysis, and influence maximization in social networks.

All epidemic models in ``FS_GPlib`` operate on **static networks** — directed or undirected graphs
with a fixed topology throughout the simulation. Both unweighted and weighted edges are supported;
when edge weights are enabled, transmission probabilities are modulated by the corresponding weights.
The models support execution on CPU and GPU, and can be combined with the batch-parallel and
distributed acceleration features described in the :doc:`/tutorial`.

Base Class & Common API
-----------------------

Every epidemic model inherits from :class:`~fs_gplib.Epidemics.base.DiffusionModel`,
which provides the shared initialisation pipeline and a uniform simulation interface.
Understanding the base class helps you leverage the full API of any concrete model.

.. autoclass:: fs_gplib.Epidemics.base.DiffusionModel
   :noindex:

Utility Methods
~~~~~~~~~~~~~~~

The following utility methods can be called on **any** epidemic model instance
to re-configure the model after construction:


.. automethod:: fs_gplib.Epidemics.base.DiffusionModel._init_node_status
   :noindex:
.. automethod:: fs_gplib.Epidemics.base.DiffusionModel._set_device
   :noindex:
.. automethod:: fs_gplib.Epidemics.base.DiffusionModel._set_seed
   :noindex:

Simulation Interface
~~~~~~~~~~~~~~~~~~~~

All epidemic models expose four progressively higher-level simulation methods.
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

   ``run_iteration`` / ``run_iterations`` do **not** reset node states, so they
   can be chained to inspect intermediate states or implement custom stopping
   criteria.  ``run_epoch`` / ``run_epochs`` always start from the initial seed
   configuration and are the recommended entry points for Monte-Carlo studies.

Available Models
----------------

The following models are available:

.. toctree::
   :maxdepth: 1

   models/epidemics/SI.rst
   models/epidemics/SIS.rst
   models/epidemics/SIR.rst
   models/epidemics/SEIR.rst
   models/epidemics/SEIR_ct.rst
   models/epidemics/SEIS.rst
   models/epidemics/SEIS_ct.rst
   models/epidemics/Threshold.rst
   models/epidemics/KThreshold.rst
   models/epidemics/IndependentCascades.rst
   models/epidemics/Profile.rst
   models/epidemics/ProfileThreshold.rst
   .. models/epidemics/SWIR.rst



