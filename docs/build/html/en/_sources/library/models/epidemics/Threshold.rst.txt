Threshold
=========

The Threshold model describes a diffusion process in which each node in a network adopts a behavior (becomes active) when a sufficient fraction or weighted sum of its neighbors are already active. Unlike the Independent Cascades model, which uses independent stochastic activations, the Threshold model relies on **deterministic or probabilistic thresholds** that control adoption.



Status
------
During the simulation, each node can be in one of two states:

+--------------+--------------+
| Status       | Code         |
+==============+==============+
| Inactive     | 0            |
+--------------+--------------+
| Active       | 1            |
+--------------+--------------+

Parameters
----------
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| Name             | Value Type                   | Default       | Mandatory | Description                                                    |
+==================+==============================+===============+===========+================================================================+
| data             | Data                         |               | Yes       | Graph data.                                                    |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| seeds            | List[int]/float in (0, 1)    |               | Yes       | Initial seed nodes or a ratio in (0, 1).                       |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| threshold        | float in [0, 1)              |               | Yes       | Adoption threshold per node (randomly generated if 0).         |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| use_weight       | Bool                         | False         | No        | Whether to use edge weights for influence aggregation.         |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| device           | 'cpu' / int (CUDA index)     | 'cpu'         | No        | Device to run the model on.                                    |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+
| rand_seed        | Int                          | None          | No        | Random seed for generating initial seeds (if ratio given).     |
+------------------+------------------------------+---------------+-----------+----------------------------------------------------------------+

Model Description
-----------------
At each iteration :math:`k`, every inactive node :math:`i` checks the fraction (or weighted fraction) of its neighbors that are active.
If this value exceeds its threshold :math:`\theta_i`, the node becomes active in the next iteration.

.. image:: ../../../images/LT-state.png
   :alt: Threshold model diagram
   :align: center
   :width: 50%

We use one Boolean indicator vector :math:`h \in \{0,1\}^N`, where :math:`h_i=1` denotes ``Active``, :math:`h_i=0` denotes ``Inactive``.

1) Each active neighbor :math:`j` of node :math:`i` transmits a contribution

.. math::
    m_{ji}^{(k)} = h_j^{(k-1)}

2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its active probability:

.. math::
    m_i^{(k)} = \frac{1}{|N(i)|}\sum_{j \in N(i)} m_{ji}^{(k)}

3) The indicator variables are updated by:

.. math::

     h_i^{(k)} =
    \begin{cases}
        1,  \text{if} (\theta_i \leq m_i^{(k)}) \land h_i^{(k-1)}=0 , \\[4pt]
        h_i^{(k-1)},  \text{otherwise},
    \end{cases} \\[6pt]


References
----------

.. [1]