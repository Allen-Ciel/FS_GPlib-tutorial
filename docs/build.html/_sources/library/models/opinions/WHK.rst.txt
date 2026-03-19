Weighted Hegselmann-Krause
==========================


The WHK (Weighted Hegselmann-Krause) model extends the classical HK model [1]_ by introducing a weight parameter :math:`w` that controls the influence strength of :math:`\varepsilon`-neighbors. Each node holds an initial opinion :math:`h \in [-1,1]` and updates it in discrete rounds. At each step:
1) :math:`\varepsilon`-neighbor identification: find the set :math:`\Gamma_\varepsilon` of each node, :math:`d_{i,j}=\left|h_i-h_j\right|\le\varepsilon`, :math:`j\in\Gamma_\varepsilon`;
2) update the opinion value: The opinion of node :math:`i` at the next time step is updated as follows:

.. math::

	h_i^{(k)}=h_i^{(k-1)}+\frac{\sum_{j\in\Gamma_\varepsilon}{h_j^{(k-1)}\cdot w_{ij}}}{\#\Gamma_\varepsilon}\cdot\left(1-\left|h_i^{(k-1)}\right|\right)

where :math:`\#\Gamma_\varepsilon` denotes the number of :math:`\varepsilon`-neighbors, and :math:`w_{ij}` is the influence weight of edge :math:`(i,j)\in E`. The factor :math:`(1-|h_i^{(k-1)}|)` ensures that opinions remain bounded in :math:`[-1,1]` and that opinions closer to the extremes are harder to shift.


Status
------
During the simulation, a node holds a continuous opinion value:

+------------+----------------+
| Status     | Range          |
+============+================+
| Opinion    | Float in [-1,1]|
+------------+----------------+

Parameters
----------
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| Name       | Type                         | Default       | Required | Description                                                               |
+============+==============================+===============+==========+===========================================================================+
| data       | Data                         |               | Yes      | Data of graph.                                                            |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| seeds      | List[float] / None           |               | Yes      | List of initial opinion values or None to generate randomly.              |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| epsilon    | float in [0, 1]              |               | Yes      | Confidence bound determining which neighbors influence the opinion update.|
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| weight     | float in [0, 1] / List[float]|               | Yes      | Influence weight applied to the neighbor average. Scalar for global       |
|            |                              |               |          | weight, or list of per-node weights.                                      |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| device     | 'cpu'/int (CUDA index)       | 'cpu'         | No       | Device to run the model on.                                               |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+
| rand_seed  | Int                          | None          | No       | Random seed for generating the initial opinion values.                    |
+------------+------------------------------+---------------+----------+---------------------------------------------------------------------------+

Implementation
--------------
The WHK model extends the HK model by (i) weighting the neighbor average with a parameter :math:`w_{ij}` and (ii) applying a bounded update rule that keeps opinions within :math:`[-1,1]`. Similar to HK, if :math:`\#\Gamma_\varepsilon=0`, the opinion remains unchanged; otherwise it is updated via the weighted aggregation.

.. image:: ../../../images/WHK-state.png
   :alt: WHK model diagram
   :align: center
   :width: 70%

1) For each neighbor :math:`j \in N(i)`, generate two message terms:

.. math::

    \begin{aligned}
        m_{ji\_c}^{(k)} &=\mathbf{1}\!\left( |h_i^{(k-1)} - h_j^{(k-1)}| < \varepsilon \right) \cdot w_{ij} \cdot h_j^{(k-1)}, \\[6pt]
        m_{ji\_v}^{(k)} &= \mathbf{1}\!\left( |h_i^{(k-1)} - h_j^{(k-1)}| < \varepsilon \right).
    \end{aligned}

where :math:`\mathbf{1}(\cdot)` is the indicator function and :math:`w_{ij}` is the influence weight of edge :math:`(i,j)\in E`.

2) Node :math:`i` aggregates received messages by computing the weighted average of :math:`\varepsilon`-neighbors:

.. math::

    \begin{aligned}
        m_i^{(k)} &=
        \begin{cases}
            \dfrac{\sum\limits_{j \in N(i)} m_{ji\_c}^{(k)}}{\sum\limits_{j \in N(i)} m_{ji\_v}^{(k)}},
            & \text{if } \sum\limits_{j \in N(i)} m_{ji\_v}^{(k)} > 0, \\[8pt]
        \mathrm{NaN}, & \text{otherwise}.
        \end{cases} \\[10pt]
    \end{aligned}


3) Finally, the opinion of node :math:`i` is updated with the bounded update rule:

.. math::

	\begin{aligned}
	h_i^{(k)} &=
	\begin{cases}
		h_i^{(k-1)} + m_i^{(k)} \cdot \left(1 - \left|h_i^{(k-1)}\right|\right), & \text{if } m_i^{(k)} \neq \mathrm{NaN}, \\[6pt]
		h_i^{(k-1)}, &  \text{otherwise}.
	\end{cases}
    \end{aligned}

The factor :math:`(1 - |h_i^{(k-1)}|)` acts as a damping term: nodes with opinions close to the boundary values :math:`\pm 1` receive smaller updates, naturally preventing the opinion from exceeding the valid range.


References
----------

.. [1] 
