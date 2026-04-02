SEIR
====

The SEIR Model [1]_ assumes that infection spreads only through links between neighboring nodes in a graph
:math:`G=(V,E)`. Each node is in one of four states: :math:`S` (susceptible), :math:`E` (exposed/incubating), :math:`I` (infected), or :math:`R` (recovered). A susceptible node
:math:`i` becomes exposed by its infected neighbors :math:`j \in N(i)` with rate :math:`\beta`, exposed nodes become infectious with rate :math:`\alpha`, and infected nodes recover with rate :math:`\gamma`:


.. math::

   \begin{aligned}
   \frac{dS_i}{dt} &= -\beta \sum_{j\in N(i)} S_i I_j, \\
   \frac{dE_i}{dt} &= \beta \sum_{j\in N(i)} S_i I_j - \alpha E_i, \\
   \frac{dI_i}{dt} &= \alpha E_i - \gamma I_i, \\
   \frac{dR_i}{dt} &= \gamma I_i .
   \end{aligned}



Implementation
--------------

Node transitions follow three rules:
1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor exposes the :math:`S` state node with probability :math:`\beta` (S→E);
2) the :math:`E` state node becomes :math:`I` with probability :math:`\alpha` (E→I);
3) the :math:`I` state node is recovered to the :math:`R` state with probability :math:`\gamma` (I→R).


.. image:: ../../../images/SEIR-state.png
   :alt: SEIR model diagram
   :align: center
   :width: 70%

Node states are represented by three Boolean indicator vectors :math:`h, e, r \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected`` or ``exposed``, :math:`h_i=0` denotes ``recovered`` or ``susceptible``,
and :math:`(h_i,e_i,r_i)=(1,0,0)` denotes ``infected``, :math:`(h_i,e_i,r_i)=(1,1,0)` denotes ``exposed``,
:math:`(h_i,e_i,r_i)=(0,0,1)` denotes ``recovered``, :math:`(h_i,e_i,r_i)=(0,0,0)` denotes ``susceptible``.
The update of the system at step :math:`k` is decomposed into three stages:

1) Each infected neighbor :math:`j` of node :math:`i` transmits a log-probability contribution for exposure (S→E)

.. math::


    m_{ji}^{(k)} = \mathbf{1}\!\left(h_j^{(k-1)}=1 \land e_j^{(k-1)}=0\right)\,\log(1-\beta)


2) Node :math:`i` collects contributions from all neighbors :math:`N(i)` to compute its exposure probability

.. math::

    m_i^{(k)} = 1 - \exp\!\left( \sum_{j \in N(i)} m_{ji}^{(k)} \right)


3) The indicator variables are updated with independent uniform random variables :math:`U_i^{\mathrm{exp}}, U_i^{\mathrm{inf}}, U_i^{\mathrm{rec}} \sim \mathrm{Uniform}(0,1)`

.. math::
    \begin{aligned}
     h_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{exp}} < m_i^{(k)}) \land (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0), \\[4pt]
        0, & \text{if } (U_i^{\mathrm{rec}} < \gamma) \land (h_i^{(k-1)}=1 \land e_i^{(k-1)}=0), \\[4pt]
        h_i^{(k-1)}, & \text{otherwise},
    \end{cases} \\[6pt]
    e_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{exp}} < m_i^{(k)}) \land (h_i^{(k-1)}=0 \land r_i^{(k-1)}=0), \\[4pt]
        0, & \text{if } (U_i^{\mathrm{inf}} < \alpha) \land e_i^{(k-1)}=1 , \\[4pt]
        e_i^{(k-1)}, & \text{otherwise},
    \end{cases} \\[6pt]
    r_i^{(k)} &=
    \begin{cases}
        1, & \text{if } (U_i^{\mathrm{rec}} < \gamma) \land (h_i^{(k-1)}=1 \land e_i^{(k-1)}=0), \\[4pt]
        r_i^{(k-1)}, & \text{otherwise}.
    \end{cases}
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
| Exposed    | 2            |
+------------+--------------+
| Recovered  | 3            |
+------------+--------------+

SEIRModel        
---------

.. autoclass:: fs_gplib.Epidemics.SEIRModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:



References
----------

.. [1] Aron J L, Schwartz I B. Seasonality and period-doubling bifurcations in an epidemic model[J]. Journal of theoretical biology, 1984, 110(4): 665-679.