SIR
===

The SIR Model [1]_:math:`^,` [2]_ assumes that infection spreads only through links between neighboring nodes in a graph
:math:`G=(V,E)`. Each node is in one of three states: :math:`S`, :math:`I`, or :math:`R`. A susceptible node
:math:`i` is infected by its infected neighbors :math:`j \in N(i)` with rate :math:`\beta`, while infected nodes recover with rate :math:`\gamma`:


.. math::

   \begin{aligned}
   \frac{dS_i}{dt} &= -\beta \sum_{j\in N(i)} S_i I_j, \\
   \frac{dI_i}{dt} &= \beta \sum_{j\in N(i)} S_i I_j - \gamma I_i, \\
   \frac{dR_i}{dt} &= \gamma I_i .
   \end{aligned}


Implementation
--------------

Node transitions follow two rules:
1) if a :math:`S` state node has :math:`I` state neighbors, each :math:`I` state neighbor transmits the infection to the :math:`S` state node with probability :math:`\beta`;
2) the :math:`I` state node is recovered to the :math:`R` state with probability :math:`\gamma`.


.. image:: ../../../images/SIR-state.png
   :alt: SIR model diagram
   :align: center
   :width: 70%

Node states are represented by two Boolean indicator vectors :math:`h, r \in \{0,1\}^N`,
where :math:`h_i=1` denotes ``infected``, :math:`r_i=1` denotes ``recovered``, and
:math:`(h_i,r_i)=(0,0)` denotes ``susceptible``.
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
		0, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} \ge \gamma), \\[4pt]
		h_i^{(k-1)}, & \text{otherwise},
	\end{cases} \\[6pt]
	r_i^{(k)} &=
	\begin{cases}
		1, & \text{if } h_i^{(k-1)}=1 \land (U_i^{\mathrm{rec}} < \gamma), \\[4pt]
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
| Recovered  | 2            |
+------------+--------------+

SIRModel
--------

.. autoclass:: fs_gplib.Epidemics.SIRModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:



References
----------

.. [1] Pastor-Satorras R, Castellano C, Van Mieghem P, et al. Epidemic processes in complex networks[J]. Reviews of modern physics, 2015, 87(3): 925-979.
.. [2] Kermack W O, McKendrick A G. A contribution to the mathematical theory of epidemics[J]. Proceedings of the royal society of london. Series A, Containing papers of a mathematical and physical character, 1927, 115(772): 700-721.


