SEIR_ct
=======

The SEIR_ct Model assumes that infection spreads only through links between neighboring nodes in a graph
:math:`G=(V,E)`. Each node is in one of four states: :math:`S` (susceptible), :math:`E` (exposed/incubating), :math:`I` (infected), or :math:`R` (recovered). Compared with the discrete-step SEIR, **SEIR_ct uses elapsed-time–dependent transition probabilities** for :math:`E\to I` and :math:`I\to R` based on the time since the last state entry.


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
| Exposed    | 2            |
+------------+--------------+
| Recovered  | 3            |
+------------+--------------+

Parameters
----------
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| Name             | Value Type                   | Default       | Mandatory | Description                                                |
+==================+==============================+===============+===========+============================================================+
| data             | Data                         |               | Yes       | Graph data.                                                |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| seeds            | List[int]/float in (0, 1)    |               | Yes       | Seed node IDs or a ratio in (0, 1).                        |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| beta             | float in [0, 1]              |               | Yes       | Infection probability per contact (S→E).                   |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| alpha            | float in [0, 1]              |               | Yes       | Incubation rate parameter (controls E→I hazard).           |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| gamma            | float in [0, 1]              |               | Yes       | Recovery rate parameter (controls I→R hazard).             |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| device           | 'cpu'/int (CUDA index)       | 'cpu'         | No        | Device to run the model on.                                |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| use_weight       | Bool                         | False         | No        | Whether to use edge weights.                               |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+
| rand_seed        | Int                          | None          | No        | Random seed for generating the initial seed set.           |
+------------------+------------------------------+---------------+-----------+------------------------------------------------------------+

Implementation
--------------

We use three Boolean indicator vectors :math:`h, e, r \in \{0,1\}^N` and two integer tensors that store **entry times**:

- :math:`h_i=1` indicates node :math:`i` is ``infected`` or ``exposed``; :math:`h_i=0` indicates ``susceptible`` or ``recovered``.
- :math:`(h_i,e_i,r_i)=(1,0,0)` denotes ``infected``, :math:`(h_i,e_i,r_i)=(1,1,0)` denotes ``exposed``.
- :math:`(h_i,e_i,r_i)=(0,0,1)` denotes ``recovered``, :math:`(h_i,e_i,r_i)=(0,0,0)` denotes ``susceptible``.
- :math:`t_i^{E}` records the iteration :math:`k` when node :math:`i` **entered E**.
- :math:`t_i^{I}` records the iteration :math:`k` when node :math:`i` **entered I**.

With :math:`k` denoting the current iteration, define duration variables
:math:`\Delta t_i^{E} = k - t_i^{E}` and :math:`\Delta t_i^{I} = k - t_i^{I}`.
The update of the system at step :math:`k` is decomposed into three stages:


.. image:: ../../../images/SEIRct-state.png
   :alt: SEIR_ct model diagram
   :align: center
   :width: 70%

1) Each infected neighbor :math:`j` of node :math:`i` transmits a log-probability contribution for exposure

.. math::


    m_{ji}^{(k)} = \mathbf{1}\!\left(h_j^{(k-1)}=1 \land e_j^{(k-1)}=0\right)\,\log(1-\beta)


2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its exposure probability

.. math::

    m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N(i)} m_{ji}^{(k)} \right)


3) The integer tensors and indicator variables are updated with independent uniform random variables :math:`U_i^{\mathrm{exp}}, U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`



.. math::
    \begin{aligned}
    t_i^E &= k, \text{if } (U_i^{\mathrm{exp}}<m_i^{(k)}) \land (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0) \\
    t_i^I &= k, \text{if } (U_i^{\mathrm{inf}}<1-exp(-\alpha \cdot \Delta t_i^E)) \land e_i^{(k-1)}=1
    \end{aligned}

.. math::
    \begin{aligned}
     h_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{exp}} < m_i^{(k)}) \land (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0), \\[4pt]
        0, & \text{if } (U_i^{\mathrm{rec}} < 1-exp(-\gamma\cdot \Delta t_i^I)) \land (h_i^{(k-1)}=1 \land e_i^{(k-1)}=0), \\[4pt]
        h_i^{(k-1)}, & \text{otherwise},
    \end{cases} \\[6pt]
    e_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{exp}} < m_i^{(k)}) \land (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0), \\[4pt]
        0, & \text{if } (U_i^{\mathrm{inf}} < 1-exp(-\alpha \cdot \Delta t_i^E)) \land e_i^{(k-1)}=1 , \\[4pt]
        e_i^{(k-1)}, & \text{otherwise},
    \end{cases} \\[6pt]
    r_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{rec}} < 1-exp(-\gamma\cdot \Delta t_i^I)) \land (h_i^{(k-1)}=1 \land e_i^{(k-1)}=0), \\[4pt]
        r_i^{(k-1)}, & \text{otherwise}.
    \end{cases}
    \end{aligned}



References
----------

.. [1]