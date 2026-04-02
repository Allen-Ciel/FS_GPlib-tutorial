DySIS
=====
The DySIS (Dynamic SIS) Model extends the classical SIS model to **time-varying networks**.
Instead of a fixed graph :math:`G=(V,E)`, diffusion evolves on a sequence of graph snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where the edge set :math:`E^{(k)}` may change at each
discrete time step :math:`k`.
Each node is in one of two states: :math:`S` (susceptible) or :math:`I` (infected).

A susceptible node :math:`i` can be infected by infected neighbors :math:`j \in N^{(k)}(i)`
with infection rate :math:`\beta`; meanwhile an infected node recovers and becomes susceptible
again with rate :math:`\lambda`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt}\bigg|_{t=k} &= -\beta \sum_{j\in N^{(k)}(i)} S_i I_j + \lambda I_i, \\
   \frac{dI_i}{dt}\bigg|_{t=k} &= \beta \sum_{j\in N^{(k)}(i)} S_i I_j - \lambda I_i .
   \end{aligned}


Compared with the static :doc:`SIS <../epidemics/SIS>` model on a single graph,
the DySIS model captures the impact of **evolving topology** on transmission and recovery dynamics.
The number of simulation steps is bounded by the length of the snapshot sequence :math:`T`.


Implementation
--------------

Node transitions follow two rules:

1) if a :math:`S` state node has :math:`I` state neighbors in the current snapshot, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) the :math:`I` state node recovers and becomes susceptible with probability :math:`\lambda`.


Node states are represented by a Boolean indicator vector :math:`h \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected`` and :math:`h_i=0` denotes ``susceptible``.
The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` in the current snapshot :math:`G^{(k)}`
transmits a log-probability contribution

.. math::

   m_{ji}^{(k)} = h_j^{(k-1)} \cdot \log\!\bigl(1-\beta\,w_{ji}^{(k)}\bigr)

where :math:`w_{ji}^{(k)}=1` if edge weights are not provided.

2) Node :math:`i` collects contributions from neighbors :math:`N^{(k)}(i)` to compute its infection probability

.. math::

   m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N^{(k)}(i)} m_{ji}^{(k)} \right)

3) The indicator variable is updated with independent uniform random variables
:math:`U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`:

.. math::

   \begin{aligned}
   h_i^{(k)} &=
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
      0, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} < \lambda), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise}.
   \end{cases}
   \end{aligned}

As in other dynamic models, :math:`N^{(k)}(i)` and optional edge weights
:math:`w_{ji}^{(k)}` are time-dependent and taken from the :math:`k`-th snapshot.
The total number of iterations is bounded by :math:`T =` ``len(edge_index_list)``.



Status
------
During the simulation, a node can be in one of the following states:

+------------+--------------+
| Status     | Code         |
+============+==============+
| Susceptible| 0            |
+------------+--------------+
| Infected   | 1            |
+------------+--------------+


DySISModel
----------

.. autoclass:: fs_gplib.Dynamic.DySISModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:


.. note::

   Unlike the static SIS model which accepts a single ``data`` object containing
   ``edge_index`` and ``edge_attr``, the DySIS model requires an explicit node
   tensor ``x`` and a **list** of edge index tensors ``edge_index_list`` representing the
   dynamic network snapshots. Edge weights are similarly provided as a list
   ``edge_attr_list``.


