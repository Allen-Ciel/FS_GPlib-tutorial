DySEIS
======

The DySEIS (Dynamic SEIS) Model extends the classical SEIS model to **time-varying networks**.
Instead of a fixed graph :math:`G=(V,E)`, diffusion evolves on a sequence of graph snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where the edge set :math:`E^{(k)}` may change at each
discrete time step :math:`k`.
Each node can be in one of three states: :math:`S` (susceptible), :math:`E` (exposed/incubating),
or :math:`I` (infected).

A susceptible node :math:`i` can be exposed by infected neighbors :math:`j \in N^{(k)}(i)`
with infection rate :math:`\beta`; exposed nodes become infectious with latent-progress rate
:math:`\alpha`; infected nodes recover back to susceptible with rate :math:`\gamma`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt}\bigg|_{t=k} &= -\beta \sum_{j\in N^{(k)}(i)} S_i I_j + \gamma I_i, \\
   \frac{dE_i}{dt}\bigg|_{t=k} &= \beta \sum_{j\in N^{(k)}(i)} S_i I_j - \alpha E_i, \\
   \frac{dI_i}{dt}\bigg|_{t=k} &= \alpha E_i - \gamma I_i .
   \end{aligned}


Compared with the static :doc:`SEIS <../epidemics/SEIS>` model on a single graph,
the DySEIS model captures the impact of **evolving topology** on exposure, infection,
and recovery dynamics. The number of simulation steps is bounded by the length of
the snapshot sequence :math:`T`.



Implementation
--------------

Node transitions follow three rules:

1) if a :math:`S` state node has :math:`I` state neighbors in the current snapshot, each :math:`I` state neighbor exposes it with probability :math:`\beta` (S→E);
2) an :math:`E` state node becomes :math:`I` with probability :math:`\alpha` (E→I);
3) an :math:`I` state node recovers to :math:`S` with probability :math:`\gamma` (I→S).

Node states are represented by two Boolean indicator vectors :math:`h, e \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected`` or ``exposed``, :math:`h_i=0` denotes ``susceptible``,
and :math:`e_i=1` denotes ``exposed``.
Therefore :math:`(h_i,e_i)=(0,0)` is ``susceptible``, :math:`(1,1)` is ``exposed``,
and :math:`(1,0)` is ``infected``.
The update at step :math:`k` is decomposed into three stages:

1) Each infected (not exposed) neighbor :math:`j` of node :math:`i` in snapshot :math:`G^{(k)}`
transmits a log-probability contribution

.. math::

   m_{ji}^{(k)} = \mathbf{1}\!\left(h_j^{(k-1)}=1 \land e_j^{(k-1)}=0\right)\,\log\!\bigl(1-\beta\,w_{ji}^{(k)}\bigr)

where :math:`w_{ji}^{(k)}=1` if edge weights are not provided.

2) Node :math:`i` aggregates contributions from neighbors :math:`N^{(k)}(i)` to obtain exposure probability

.. math::

   m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N^{(k)}(i)} m_{ji}^{(k)} \right)

3) Indicator variables are updated with independent uniform random variables
:math:`U_i^{\mathrm{exp}}, U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`:

.. math::

   \begin{aligned}
   h_i^{(k)} &=
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{exp}} < m_i^{(k)}), \\[4pt]
      0, & \text{if } (h_i^{(k-1)}=1 \land e_i^{(k-1)}=0) \land (U_i^{\mathrm{rec}} < \gamma), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise},
   \end{cases} \\[6pt]
   e_i^{(k)} &=
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{exp}} < m_i^{(k)}), \\[4pt]
      0, & \text{if } e_i^{(k-1)}=1 \land (U_i^{\mathrm{inf}} < \alpha), \\[4pt]
      e_i^{(k-1)}, & \text{otherwise}.
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
| Exposed    | 2            |
+------------+--------------+

DySEISModel
-----------


.. autoclass:: fs_gplib.Dynamic.DySEISModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:


.. note::

   Unlike the static SEIS model which accepts a single ``data`` object containing
   ``edge_index`` and ``edge_attr``, the DySEIS model requires an explicit node
   tensor ``x`` and a **list** of edge index tensors ``edge_index_list`` representing
   dynamic network snapshots. Edge weights are similarly provided as a list
   ``edge_attr_list``.

