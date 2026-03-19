Sznajd
======


The Sznajd model [1]_ is a binary opinion dynamics model inspired by the principle "United we stand, divided we fall." Each node holds a binary opinion :math:`h \in \{0, 1\}`. At each time step:

1) a node :math:`u` is selected uniformly at random from the network;
2) a neighbor :math:`v \in N(u)` of node :math:`u` is selected uniformly at random;
3) if :math:`u` and :math:`v` share the same opinion (:math:`h_u = h_v`), all neighbors of both :math:`u` and :math:`v` adopt that shared opinion;
4) if :math:`u` and :math:`v` disagree (:math:`h_u \neq h_v`), no opinion change occurs.



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
The Sznajd model propagates consensus along agreed-upon edges: when two connected nodes share the same opinion, they jointly persuade all of their neighbors to adopt that opinion. Self-loops are removed to prevent a node from influencing itself. The model uses max aggregation to broadcast the agreed opinion to the neighborhoods of the two concordant nodes.


.. image:: ../../../images/Sznajd-state.png
   :alt: Sznajd model diagram
   :align: center
   :width: 70%

1) A node :math:`u` is selected uniformly at random. A neighbor :math:`v \in N(u)` is chosen uniformly at random.

2) If :math:`u` and :math:`v` agree (:math:`h_u^{(k-1)} = h_v^{(k-1)}`), construct an auxiliary signal vector over all nodes:

.. math::

    x_j =
    \begin{cases}
        h_u^{(k-1)}, & \text{if } j \in \{u, v\}, \\[6pt]
        -1, & \text{otherwise}.
    \end{cases}

3) Propagate :math:`x` over the graph using max aggregation. Each node receives the maximum signal among its neighbors:

.. math::

    m_j^{(k)} = \max_{i \in N(j)}\, x_i

Nodes adjacent to :math:`u` or :math:`v` will receive :math:`h_u^{(k-1)} \in \{0,1\}`, while all other nodes receive :math:`-1`.

4) Update opinions for nodes that received a valid signal:

.. math::

    h_j^{(k)} =
    \begin{cases}
        m_j^{(k)}, & \text{if } m_j^{(k)} \neq -1, \\[6pt]
        h_j^{(k-1)}, & \text{otherwise}.
    \end{cases}

If :math:`u` and :math:`v` disagree (:math:`h_u^{(k-1)} \neq h_v^{(k-1)}`), all opinions remain unchanged: :math:`h_j^{(k)} = h_j^{(k-1)}` for every node :math:`j`.


References
----------

.. [1] 