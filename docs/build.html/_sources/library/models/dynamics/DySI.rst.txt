DySI
====
The DySI (Dynamic SI) Model extends the classical SI model to **time-varying networks**.
Instead of a fixed graph :math:`G=(V,E)`, the system evolves on a sequence of graph snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where the edge set :math:`E^{(k)}` changes at each
discrete time step :math:`k`.
Each node is in one of two states: :math:`S` (susceptible) or :math:`I` (infected).
A susceptible node :math:`i` becomes infected by its infected neighbors
:math:`j \in N^{(k)}(i)` in the current snapshot with rate :math:`\beta`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt}\bigg|_{t=k} &= -\beta \sum_{j\in N^{(k)}(i)} S_i I_j, \\
   \frac{dI_i}{dt}\bigg|_{t=k} &= \beta \sum_{j\in N^{(k)}(i)} S_i I_j .
   \end{aligned}


Compared with the static :doc:`SI <../epidemics/SI>` model which operates on a single fixed graph,
the DySI model captures the influence of **evolving network topology** on infection dynamics.
The number of simulation steps is bounded by the length of the snapshot sequence :math:`T`.


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

Parameters
----------
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| Name           | Value Type                   | Default       | Mandatory | Description                                      |
+================+==============================+===============+===========+==================================================+
| x              | Tensor                       |               | Yes       | Node tensor of shape :math:`(N, 1)`.             |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| edge_index_list| List[Tensor]                 |               | Yes       | List of edge index tensors, one per snapshot.    |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| seeds          | List[int]/float in (0, 1)    |               | Yes       | List of seed node IDs or a ratio in (0, 1).      |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| infection_beta | float in [0, 1]              |               | Yes       | Infection probability.                           |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| device         | 'cpu'/int (CUDA index)       | 'cpu'         | No        | Device to run the model on.                      |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| edge_attr_list | List[Tensor]                 | None          | No        | List of edge weight tensors, one per snapshot.   |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| rand_seed      | Int                          | None          | No        | Random seed for generating the seed set.         |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+

.. note::

   Unlike the static SI model which accepts a single ``data`` object containing
   ``edge_index`` and ``edge_attr``, the DySI model requires an explicit node
   tensor ``x`` and a **list** of edge index tensors ``edge_index_list`` representing the
   dynamic network snapshots. Edge weights are similarly provided as a list
   ``edge_attr_list`` instead of a single ``use_weight`` flag.


Implementation
--------------

Node transitions follow two rules:

1) if a :math:`S` state node has :math:`I` state neighbors **in the current snapshot** :math:`G^{(k)}`, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) once infected, the node remains infected forever.

Node states are represented by a Boolean indicator vector :math:`h \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected`` and :math:`h_i=0` denotes ``susceptible``.
The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` **in the current snapshot** transmits a log-probability contribution

.. math::

   m_{ji}^{(k)} = h_j^{(k-1)} \cdot \log\!\bigl(1-\beta\bigr)


2) Node :math:`i` collects contributions from all neighbors :math:`N^{(k)}(i)` to compute its infection probability

.. math::

   m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N^{(k)}(i)} m_{ji}^{(k)} \right)

3) The indicator variable is updated with independent uniform random variables :math:`U_i^{\mathrm{inf}} \sim \mathrm{Uniform}(0,1)`:

.. math::

   h_i^{(k)} =
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise}.
   \end{cases}

The key distinction from the static SI model is that the neighbor set :math:`N^{(k)}(i)` and the
edge weights :math:`w_{ji}^{(k)}` (If provided) are **time-dependent**, drawn from the :math:`k`-th graph snapshot.
The total number of iterations is bounded by the number of available snapshots :math:`T =` ``len(edge_index_list)``.


References
----------

.. [1] 
