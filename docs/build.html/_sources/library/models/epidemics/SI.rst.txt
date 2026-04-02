SI
==
The SI Model [1]_ assumes that infection spreads only through links between neighboring nodes in a graph
:math:`G=(V,E)`. Each node is in one of two states: :math:`S` (susceptible) or :math:`I` (infected).
A susceptible node :math:`i` becomes infected by its infected neighbors :math:`j \in N(i)` with rate :math:`\beta`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt} &= -\beta \sum_{j\in N(i)} S_i I_j, \\
   \frac{dI_i}{dt} &= \beta \sum_{j\in N(i)} S_i I_j .
   \end{aligned}

Implementation
--------------

Node transitions follow two rules:
1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) once infected, the node remains infected forever.

.. image:: ../../../images/SI-state.png
   :alt: SI model diagram
   :align: center
   :width: 40%

Node states are represented by a Boolean indicator vector :math:`h \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected`` and :math:`h_i=0` denotes ``susceptible``.
The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` transmits a log-probability contribution

.. math::

   m_{ji}^{(k)} = h_j^{(k-1)} \cdot \log(1-\beta)

2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its infection probability

.. math::

   m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N(i)} m_{ji}^{(k)} \right)

3) The indicator variable is updated with independent uniform random variables :math:`U_i^{\mathrm{inf}} \sim \mathrm{Uniform}(0,1)`:

.. math::

   h_i^{(k)} =
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise}.
   \end{cases}

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

SIModel
-------

.. autoclass:: fs_gplib.Epidemics.SIModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:


References
----------

.. [1] Kermack W O, McKendrick A G. A contribution to the mathematical theory of epidemics[J]. Proceedings of the royal society of london. Series A, Containing papers of a mathematical and physical character, 1927, 115(772): 700-721.