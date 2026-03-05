Independent Cascades
====================

The Independent Cascades (IC) model describes information diffusion or
contagion processes on a static graph :math:`G=(V,E)`,
where each node can be either **active (infected and recovered)** or **inactive (susceptible)**.
Once a node becomes active, it gets **one chance** to activate each of its inactive neighbors with a given probability (**infected**);
whether or not it succeeds, it has no more chance to activate others (**recovered**).


Status
------
During the simulation, a node can be in one of the following states:

+--------------+--------------+
| Status       | Code         |
+==============+==============+
| Susceptible  | 0            |
+--------------+--------------+
| Infected     | 1            |
+--------------+--------------+

Parameters
----------
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| Name             | Value Type                   | Default       | Mandatory | Description                                                |
+==================+==============================+===============+===========+============================================================+
| data             | Data                         |               | Yes       | Graph data.                                                |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| seeds            | List[int]/float in (0, 1)    |               | Yes       | Seed node IDs or a ratio in (0, 1).                        |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| beta             | float in [0, 1]              |               | Yes       | Infection probability per contact (activation probability).|
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| device           | 'cpu' / int (CUDA index)     | 'cpu'         | No        | Device to run the model on.                                |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| use_weight       | Bool                         | False         | No        | Whether to use edge weights.                               |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| rand_seed        | Int                          | None          | No        | Random seed for generating the initial seed set.           |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+

Implementation
--------------
Independent Cascades is the special case of the SIR Model, where :math: `\gamma=1`.

Node transitions follow two rules:
1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) the :math:`I` state node is recovered to the :math:`R` state with probability :math:`1`.


.. image:: ../../../images/IC-state.png
   :alt: Independent Cascades model diagram
   :align: center
   :width: 70%


Node states are represented by two Boolean indicator vectors :math:`h, r \in \{0,1\}^N`:

- :math:`h_i=1` denotes ``infected`` (**Active**)
- :math:`r_i=1` denotes ``recovered`` (**Active**)
- :math:`(h_i,r_i)=(0,0)` denotes ``susceptible`` (**Inactive**)

The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` transmits a log-probability contribution

.. math::

	m_{ji}^{(k)} = h_j^{(k-1)} \cdot \log(1-\beta)


2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its infection probability

.. math::

	m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N(i)} m_{ji}^{(k)} \right)


3) The indicator variables are updated with independent uniform random variables :math:`U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`

.. math::

    \begin{aligned}
	h_i^{(k)} &=
	\begin{cases}
		1, & \text{if } (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0) \land (U_i^{\mathrm{inf}} < m_i^{(k)}), \\[4pt]
		0, & \text{if } h_i^{(k-1)}=1, \\[4pt]
		h_i^{(k-1)}, & \text{otherwise},
	\end{cases} \\[6pt]
	r_i^{(k)} &=
	\begin{cases}
		1, & \text{if } h_i^{(k-1)}=1, \\[4pt]
		r_i^{(k-1)}, & \text{otherwise}.
	\end{cases}
    \end{aligned}

References
----------

.. [1]
