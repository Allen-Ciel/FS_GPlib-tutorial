Q-Voter
=======


The Q-Voter model [1]_ is a nonlinear extension of the classical Voter model that incorporates social reinforcement and stochastic independence. Each node holds a binary opinion :math:`h \in \{0, 1\}`. At each time step:

1) a node :math:`i` is selected uniformly at random from the network;
2) :math:`q` neighbors of node :math:`i` are independently sampled (with replacement);
3) if all :math:`q` sampled neighbors share the same opinion, node :math:`i` adopts that unanimous opinion;
4) otherwise, with probability :math:`\epsilon` (independence), node :math:`i` flips its current opinion; with probability :math:`1-\epsilon`, it retains its current opinion.


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
| q          | Int (> 0)                    |               | Yes      | Number of influence samples (social reinforcement group size).            |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| epsilon    | float in [0, 1]              |               | Yes      | Independence (noise) probability for non-unanimous interactions.          |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| device     | 'cpu'/int (CUDA index)       | 'cpu'         | No       | Device to run the model on.                                               |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| rand_seed  | Int                          | None          | No       | Random seed for reproducibility.                                          |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+

Implementation
--------------
The Q-Voter model extends the Voter model by introducing a nonlinear social reinforcement mechanism: a group of :math:`q` sampled opinions must reach unanimous agreement before the focal node conforms. When unanimity is not achieved, the independence parameter :math:`\epsilon` governs the probability of a random opinion flip, preventing the system from freezing. Self-loops are removed to prevent a node from influencing itself.

.. image:: ../../../images/QVoter-state.png
   :alt: Q-Voter model diagram
   :align: center
   :width: 70%

1) For each neighbor :math:`j \in N(i)`, generate a message equal to its current opinion:

.. math::

    m_{ji}^{(k)} = h_j^{(k-1)}

2) Node :math:`i` aggregates received messages by computing the mean opinion of its neighbors:

.. math::

    m_i^{(k)} = \frac{\sum_{j \in N(i)} m_{ji}^{(k)}}{\#N(i)}

3) A node index :math:`i` is selected uniformly at random. Draw :math:`q` independent samples :math:`b_l \sim \mathrm{Bernoulli}(m_i^{(k)})` for :math:`l = 1, \ldots, q`. Its opinion is updated as follows:

.. math::

	h_i^{(k)} =
	\begin{cases}
		b_1, & \text{if } b_1 = b_2 = \cdots = b_q, \\[6pt]
		1 - h_i^{(k-1)}, & \text{if } b_l \text{'s are not all equal and } U_i < \epsilon, \\[6pt]
		h_i^{(k-1)}, & \text{otherwise},
	\end{cases}

where :math:`U_i \sim \mathrm{Uniform}(0,1)` is an independent random number and :math:`1 - h_i^{(k-1)}` denotes the flip of the current opinion. Drawing :math:`q` samples from :math:`\mathrm{Bernoulli}(m_i^{(k)})` is statistically equivalent to sampling :math:`q` neighbors with replacement and reading their opinions. When :math:`q = 1`, the single sample is always trivially unanimous and the model reduces to the standard Voter model. As :math:`q` increases, unanimous agreement becomes harder to achieve, increasing the role of the independence parameter :math:`\epsilon`. All other nodes retain their current opinions.

References
----------

.. [1]
