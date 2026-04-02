DyKThreshold
============

The DyKThreshold (Dynamic Kertesz Threshold) model extends the static
:doc:`KThreshold <../epidemics/KThreshold>` model to **time-varying networks**.
Instead of evolving on a fixed graph :math:`G=(V,E)`, diffusion is performed on
a sequence of snapshots
:math:`\{G^{(k)}=(V,E^{(k)})\}_{k=1}^{T}`, where neighborhood relations may
change at each discrete step :math:`k`.

Compared with DyThreshold, DyKThreshold introduces two extra mechanisms:

1) **spontaneous adoption** with probability ``adopter_rate``;
2) **blocked nodes** sampled once from initially inactive nodes with ratio
   ``percentage_blocked``.

Blocked nodes never become active. Non-blocked inactive nodes can become active
either by spontaneous adoption or by threshold-based adoption.



Implementation
--------------

Let :math:`h_i^{(k)} \in \{0,1\}` denote the activity indicator of node
:math:`i` at step :math:`k`, and :math:`b_i \in \{0,1\}` denote whether node
:math:`i` is blocked.

- :math:`h_i=0, b_i=0`: inactive
- :math:`h_i=1, b_i=0`: active
- :math:`h_i=0, b_i=1`: blocked

The dynamics follow four stages:

1) **Blocked-node initialization** (performed once):
   among initially inactive nodes, sample a subset as blocked according to
   ``percentage_blocked``.

2) **Neighbor contribution** at snapshot :math:`k`:

.. math::

   m_{ji}^{(k)} = w_{ji}^{(k)} h_j^{(k-1)},

where :math:`w_{ji}^{(k)}=1` when edge weights are not provided.

3) **Influence aggregation** for node :math:`i`:

.. math::

   m_i^{(k)} = \operatorname{Agg}_{j\in N^{(k)}(i)} m_{ji}^{(k)}.

When ``edge_attr_list`` is not provided, aggregation uses mean influence;
otherwise weighted sum is used.

4) **State update** for non-blocked inactive nodes:

.. math::

   h_i^{(k)} =
   \begin{cases}
      1, & \text{if } h_i^{(k-1)}=0 \land b_i=0 \land
      \bigl((U_i^{\mathrm{adopt}} < \alpha)\ \lor\ (\theta_i \le m_i^{(k)})\bigr), \\[4pt]
      h_i^{(k-1)}, & \text{otherwise},
   \end{cases}

where :math:`U_i^{\mathrm{adopt}} \sim \mathrm{Uniform}(0,1)`,
:math:`\alpha =` ``adopter_rate``, and :math:`\theta_i` is the node threshold.

Blocked nodes are fixed and always reported as state ``-1`` in outputs.
As with other dynamic models, :math:`N^{(k)}(i)` and optional edge weights are
time-dependent, and the maximum number of iterations is bounded by
:math:`T =` ``len(edge_index_list)``.


Status
------
During the simulation, a node can be in one of the following states:

+-----------+--------------+
| Status    | Code         |
+===========+==============+
| Inactive  | 0            |
+-----------+--------------+
| Active    | 1            |
+-----------+--------------+
| Blocked   | -1           |
+-----------+--------------+

DyKerteszThresholdModel
-----------------------

.. autoclass:: fs_gplib.Dynamic.DyKerteszThresholdModel
   :members: run_iteration, run_iterations, run_epoch, run_epochs
   :member-order: bysource
   :show-inheritance:

Parameters
----------
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| Name               | Value Type                   | Default       | Mandatory | Description                                          |
+====================+==============================+===============+===========+======================================================+
| x                  | Tensor                       |               | Yes       | Node tensor of shape :math:`(N, 1)`.                 |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| edge_index_list    | List[Tensor]                 |               | Yes       | List of edge index tensors, one per snapshot.        |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| seeds              | List[int]/float in (0, 1)    |               | Yes       | Initial active node IDs or a ratio.                  |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| threshold          | float in [0, 1]              |               | Yes       | Node threshold. If 0, random thresholds are sampled. |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| adopter_rate       | float in [0, 1]              |               | Yes       | Spontaneous adoption probability per step.           |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| percentage_blocked | float in [0, 1]              |               | Yes       | Ratio of blocked nodes among initially inactive.     |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| device             | 'cpu'/int (CUDA index)       | 'cpu'         | No        | Device to run the model on.                          |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| edge_attr_list     | List[Tensor]                 | None          | No        | List of edge weight tensors, one per snapshot.       |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+
| rand_seed          | Int                          | None          | No        | Random seed for generating the seed set.             |
+--------------------+------------------------------+---------------+-----------+------------------------------------------------------+

.. note::

   The dynamic model uses ``x`` + ``edge_index_list`` instead of a single
   static ``data`` object. Optional edge weights are passed as
   ``edge_attr_list``.

