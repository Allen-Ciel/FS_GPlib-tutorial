Kertesz Threshold
=================

The Kertesz Threshold [1]_ model extends the classical Threshold model by introducing two additional mechanisms: **spontaneous adoption** and **node blocking**. A fraction of susceptible nodes are randomly designated as *Blocked* at initialization, permanently preventing their activation. Non-blocked inactive nodes can become active either by spontaneously adopting with a given probability, or through the standard threshold mechanism when sufficient neighbor influence is accumulated.




Model Description
-----------------

Node transitions follow three rules:

1. When :math:`k=0`, a fraction :math:`p` (``percentage_blocked``) of susceptible (inactive) nodes are randomly designated as **Blocked**. Blocked nodes can never become active.

2. For :math:`k>0`, every non-blocked inactive node :math:`i` may be activated through one of two mechanisms:

    **Mechanism 1: Spontaneous adoption.** Each non-blocked inactive node :math:`i` independently adopts with probability :math:`\alpha` (``adopter_rate``):

    **Mechanism 2: Threshold-based adoption.** Every inactive node :math:`i` checks the fraction (or weighted fraction) of its neighbors that are active. If this value exceeds its threshold :math:`\theta_i`, the node becomes active in the next iteration.




.. image:: ../../../images/KT-state.png
   :alt: Kertesz Threshold model diagram
   :align: center
   :width: 50%


We use two indicator vectors: :math:`h, b \in \{0,1\}^N` to represent node states:

- :math:`h_i=1` and :math:`b_i=0` denotes **Active**,
- :math:`h_i=0` and :math:`b_i=0` denotes **Inactive**,
- :math:`h_i=0` and :math:`b_i=1` denotes **Blocked**.


The update of the system at step :math:`k` is decomposed into three stages:


1) Each active neighbor :math:`j` of node :math:`i` transmits a contribution

.. math::
    m_{ji}^{(k)} = h_j^{(k-1)}

2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its active probability:

.. math::
    m_i^{(k)} = \frac{1}{|N(i)|}\sum_{j \in N(i)} m_{ji}^{(k)} 

3) The indicator variables are updated with independent uniform random variables :math:`U_i \sim \mathrm{Uniform}(0,1)`:

.. math::

     h_i^{(k)} =
    \begin{cases}
        1,  \text{if } (h_i^{(k-1)}=0 \land b_i=0) \land ((U_i < \alpha) \lor (\theta_i \leq m_i^{(k)})), \\[4pt]
        h_i^{(k-1)},  \text{otherwise},
    \end{cases} \\[6pt]

Status
------
During the simulation, each node can be in one of three states:

+--------------+--------------+
| Status       | Code         |
+==============+==============+
| Inactive     | 0            |
+--------------+--------------+
| Active       | 1            |
+--------------+--------------+
| Blocked      | -1           |
+--------------+--------------+


KerteszThresholdModel
---------------------

.. autoclass:: fs_gplib.Epidemics.KerteszThresholdModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:



References
----------

.. [1] Ruan Z, Iniguez G, Karsai M, et al. Kinetics of social contagion[J]. Physical review letters, 2015, 115(21): 218702.