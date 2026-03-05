SIS
===


The SIS Model [1]_ assumes that infection spreads only through links between neighboring nodes in a graph
:math:`G=(V,E)`. Each node is in one of two states: :math:`S` (susceptible) or :math:`I` (infected).
A susceptible node :math:`i` becomes infected by its infected neighbors :math:`j \in N(i)` with rate :math:`\beta`,
while infected nodes recover and become susceptible again with rate :math:`\gamma`:

.. math::

   \begin{aligned}
   \frac{dS_i}{dt} &= -\beta \sum_{j\in N(i)} S_i I_j + \gamma I_i, \\
   \frac{dI_i}{dt} &= \beta \sum_{j\in N(i)} S_i I_j - \gamma I_i .
   \end{aligned}


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
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| Name       | Value Type                   | Default       | Mandatory | Description                                |
+============+==============================+===============+===========+============================================+
| data       | Data                         |               | Yes       | Data of graph.                             |
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| seeds      | List[int]/float in (0, 1)    |               | Yes       | List of seed node IDs or a ratio in (0, 1).|
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| beta       | float in [0, 1]              |               | Yes       | Infection probability.                     |
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| gamma      | float in [0, 1]              |               | Yes       | Recovery probability.                      |
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| device     | 'cpu'/int (CUDA index)       | 'cpu'         | No        | Device to run the model on.                |
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| use_weight | Bool                         | False         | No        | Whether to use edge weights.               |
+------------+------------------------------+---------------+-----------+--------------------------------------------+
| rand_seed  | Int                          | None          | No        | Random seed for generating the seed set.   |
+------------+------------------------------+---------------+-----------+--------------------------------------------+

Implementation
--------------

Node transitions follow two rules:
1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) the :math:`I` state node recovers and becomes susceptible again with probability :math:`\gamma`.

.. image:: ../../../images/SIS-state.png
   :alt: SIS model diagram
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


3) The indicator variable is updated with independent uniform random variables
:math:`U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`:

.. math::

   \begin{aligned}
   h_i^{(k)} &=
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
      0, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} < \gamma), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise}.
   \end{cases}
   \end{aligned}

References
----------

.. [1]