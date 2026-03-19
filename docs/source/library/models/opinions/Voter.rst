Voter
=====


The Voter model [1]_ is a classical binary opinion dynamics model. Each node holds a discrete opinion :math:`s \in \{0, 1\}`. At each time step:
1) a node :math:`i` is selected uniformly at random from the network;
2) node :math:`i` adopts the opinion of a randomly chosen neighbor :math:`j \in N(i)`.

This process is equivalent to computing the fraction of neighbors holding opinion 1, then updating the selected node's opinion by sampling a Bernoulli random variable with that fraction as the success probability:

.. math::

	s_i^{(k)} \sim \mathrm{Bernoulli}\!\left( \frac{\sum_{j \in N(i)} s_j^{(k-1)}}{\#N(i)} \right)

where :math:`\#N(i)` denotes the number of neighbors of node :math:`i`.


Status
------
During the simulation, a node holds a binary opinion value:

+------------+----------------+
| Status     | Value          |
+============+================+
| Opinion    | 0 or 1         |
+------------+----------------+

Parameters
----------
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| Name       | Type                         | Default       | Required | Description                                                               |
+============+==============================+===============+==========+===========================================================================+
| data       | Data                         |               | Yes      | Data of graph.                                                            |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| seeds      | List[int] / None             |               | Yes      | List of initially activated node indices (opinion = 1) or None.           |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| device     | 'cpu'/int (CUDA index)       | 'cpu'         | No       | Device to run the model on.                                               |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| rand_seed  | Int                          | None          | No       | Random seed for reproducibility.                                          |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+

Implementation
--------------
The Voter model uses binary opinions, so each node's state is simply 0 or 1. At each iteration a single randomly chosen node updates its opinion via neighbor aggregation. Self-loops are removed to prevent a node from influencing itself.

.. image:: ../../../images/Voter-state.png
   :alt: Voter model diagram
   :align: center
   :width: 70%

1) For each neighbor :math:`j \in N(i)`, generate a message equal to its current opinion:

.. math::

    m_{ji}^{(k)} = s_j^{(k-1)}

2) Node :math:`i` aggregates received messages by computing the mean opinion of its neighbors:

.. math::

    m_i^{(k)} = \frac{\sum_{j \in N(i)} m_{ji}^{(k)}}{\#N(i)}

3) A node index :math:`i` is selected uniformly at random. Its opinion is updated by sampling:

.. math::

	s_i^{(k)} =
	\begin{cases}
		1, & \text{if } U_i < m_i^{(k)}, \\[6pt]
		0, & \text{otherwise},
	\end{cases}

where :math:`U_i \sim \mathrm{Uniform}(0,1)` is a random number. All other nodes retain their current opinions.


References
----------

.. [1] 
