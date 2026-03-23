DySIR
=====
The DySIR (Dynamic SIR) Model extends the classical SIR model to **time-varying networks**.
Instead of a fixed graph :math:`G=(V,E)`, diffusion evolves on a sequence of graph snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where the edge set :math:`E^{(k)}` may change at each
discrete time step :math:`k`.
Each node can be in one of three states: :math:`S` (susceptible), :math:`I` (infected), or :math:`R` (removed/recovered).

A susceptible node :math:`i` can be infected by infected neighbors :math:`j \in N^{(k)}(i)`
with infection rate :math:`\beta`; meanwhile an infected node recovers independently with
rate :math:`\lambda`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt}\bigg|_{t=k} &= -\beta \sum_{j\in N^{(k)}(i)} S_i I_j, \\
   \frac{dI_i}{dt}\bigg|_{t=k} &= \beta \sum_{j\in N^{(k)}(i)} S_i I_j - \lambda I_i, \\
   \frac{dR_i}{dt}\bigg|_{t=k} &= \lambda I_i .
   \end{aligned}


Compared with the static :doc:`SIR <../epidemics/SIR>` model on a single graph,
the DySIR model captures the impact of **evolving topology** on transmission and recovery dynamics.
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
| Removed    | 2            |
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
| recovery_lambda| float in [0, 1]              |               | Yes       | Recovery probability for infected nodes.         |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| device         | 'cpu'/int (CUDA index)       | 'cpu'         | No        | Device to run the model on.                      |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| edge_attr_list | List[Tensor]                 | None          | No        | List of edge weight tensors, one per snapshot.   |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+
| rand_seed      | Int                          | None          | No        | Random seed for generating the seed set.         |
+----------------+------------------------------+---------------+-----------+--------------------------------------------------+

.. note::

   Unlike the static SIR model which accepts a single ``data`` object containing
   ``edge_index`` and ``edge_attr``, the DySIR model requires an explicit node
   tensor ``x`` and a **list** of edge index tensors ``edge_index_list`` representing the
   dynamic network snapshots. Edge weights are similarly provided as a list
   ``edge_attr_list``.


Implementation
--------------

Node transitions follow three rules:

1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) the :math:`I` state node is recovered to the :math:`R` state with probability :math:`\lambda`;
3) once recovered, a node remains in :math:`R`.



Node states are represented by two Boolean indicator vectors :math:`h, r \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected``, :math:`r_i=1` denotes ``removed/recovered``, and
:math:`(h_i,r_i)=(0,0)` denotes ``susceptible``.
The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` in the current snapshot :math:`G^{(k)}`
transmits a log-probability contribution

.. math::

   m_{ji}^{(k)} = h_j^{(k-1)} \cdot \log\!\bigl(1-\beta\,w_{ji}^{(k)}\bigr)

where :math:`w_{ji}^{(k)}=1` if edge weights are not provided.

2) Node :math:`i` collects contributions from neighbors :math:`N^{(k)}(i)` to compute its infection probability

.. math::

   m_i^{(k)} = \left(1-r_i^{(k-1)}\right)\!\left[1 - \exp\!\left( \sum_{j \in N^{(k)}(i)} m_{ji}^{(k)} \right)\right]

3) The indicator variables are updated with independent uniform random variables
:math:`U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`

.. math::

    \begin{aligned}
	h_i^{(k)} &=
	\begin{cases}
		1, & \text{if } (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0) \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
		0, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} < \lambda), \\[4pt]
		h_i^{(k-1)}, & \text{otherwise},
	\end{cases} \\[6pt]
	r_i^{(k)} &=
	\begin{cases}
		1, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} < \lambda), \\[4pt]
		r_i^{(k-1)}, & \text{otherwise}.
	\end{cases}
    \end{aligned}


As in other dynamic models, :math:`N^{(k)}(i)` and optional edge weights
:math:`w_{ji}^{(k)}` are time-dependent and taken from the :math:`k`-th snapshot.
The total number of iterations is bounded by :math:`T =` ``len(edge_index_list)``.


References
----------

.. [1]