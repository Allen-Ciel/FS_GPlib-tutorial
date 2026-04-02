Advanced
========

Ultra-Large Graph Propagation
-----------------------------

For graphs that are too large to fit comfortably on a single device, the main
challenge is usually not the propagation rule itself, but the storage and
movement of graph structure and node states.

In ``FS_GPlib``, the built-in epidemic, opinion, and dynamic models already
provide efficient tensor-based propagation and batch-parallel Monte-Carlo
simulation.  However, for **web-scale** or otherwise extremely large graphs,
users often need an additional layer of graph partitioning so that simulation
can be organised across multiple processes, GPUs, or machines.

The library therefore provides an advanced **graph partitioning interface** for 
users.  It does **not** provide a fully integrated distributed simulation API.

What the Library Provides
~~~~~~~~~~~~~~~~~~~~~~~~~

``FS_GPlib`` currently provides:

- a graph partitioner that splits a static graph into target-node-owned
  subgraphs;
- a loader for reading saved partitions back from disk.

``FS_GPlib`` does not currently provide:

- a ready-to-use distributed ``run_epochs`` API;
- built-in process-group management;
- automatic cross-partition synchronisation of node states;
- launch scripts for multi-GPU or multi-machine execution.

This design keeps the algorithm library lightweight while still exposing the
key primitive needed to build ultra-large-scale propagation workflows.

Why Partition by Target Nodes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most propagation models in ``FS_GPlib`` are implemented with message passing.
At each step, a target node aggregates information from incoming neighbors.
Because of that, a natural distributed decomposition is:

- assign each process a subset of **target nodes**;
- keep all edges whose destination lies in that subset;
- compute updates for only those owned target nodes;
- synchronise node states across processes after each step if needed.

This target-node decomposition has two practical benefits:

- the per-partition workload is aligned with the aggregation direction used by
  ``edge_index[1]``;
- each process only needs to own the edges that contribute to its target-node
  updates.

At the same time, note that source nodes of those edges may belong to other
partitions, so the node states still need to be synchronised externally by the
user's distributed loop.

Partitioning Strategy
~~~~~~~~~~~~~~~~~~~~~

The partitioner in ``fs_gplib.DistributedComputing`` uses the number of edges
ending at each target node as a simple load proxy, and applies a
Longest Processing Time (LPT) heuristic to obtain approximately balanced
target-node groups:

1. count the incoming edges of each destination node;
2. sort destination nodes by edge count in descending order;
3. split the ordered nodes into several groups according to cumulative edge
   volume;
4. for each group, keep only the edges whose destination belongs to that
   group.

This strategy aims to reduce skew between partitions when node in-degrees are
highly uneven.

Data Layout After Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After partitioning, each partition contains:

- ``sub_data``: a PyG ``Data`` object whose ``edge_index`` contains only edges
  ending at the owned destination nodes;
- ``sub_targets``: the destination nodes owned by that partition.

An important detail is that the filtered subgraph keeps the **original global
node indexing**.  In other words, the graph is edge-filtered rather than
relabelled to a local node id space.  This makes it easier to merge or
synchronise node states across partitions during simulation.

Recommended Workflow
~~~~~~~~~~~~~~~~~~~~

For ultra-large graph simulation, the typical workflow is:

1. prepare a global PyG ``Data`` object;
2. partition the graph once and optionally save partitions to disk;
3. launch your own distributed or multi-process program;
4. in each worker, load one partition;
5. build the desired propagation model on that partition;
6. update owned target nodes locally and synchronise node states between
   workers after each iteration.

If you want to see a concrete distributed simulation example, refer to the
:doc:`/tutorial`.  The present page focuses only on the reusable partitioning
API and the design considerations behind it.

Distributed Partitioning API
----------------------------

.. currentmodule:: fs_gplib.DistributedComputing.DistributedComputing

GraphPartitioner
~~~~~~~~~~~~~~~~

.. autoclass:: GraphPartitioner
   :members: generate_partition
   :member-order: bysource

Usage Notes
~~~~~~~~~~~

- ``GraphPartitioner(data, n_parts, root=None)`` creates the partitioner.
- If ``root`` is ``None``, partitions are kept in memory in ``self.sub_data``.
- If ``root`` is given, partitions are written to ``root/part_i/``.
- Each saved partition contains ``sub_data.pt`` and ``sub_nodes.pt``.

In-memory example:

.. code-block:: python

   from fs_gplib.DistributedComputing.DistributedComputing import GraphPartitioner

   partitioner = GraphPartitioner(data, n_parts=4)
   partitioner.generate_partition()

   sub_data_0 = partitioner.sub_data[0]
   sub_targets_0 = partitioner.sub_nodes[0]

Save-to-disk example:

.. code-block:: python

   from fs_gplib.DistributedComputing.DistributedComputing import GraphPartitioner

   partitioner = GraphPartitioner(data, n_parts=4, root="./dataset/graph_parts")
   partitioner.generate_partition()

Loading Saved Partitions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_partition

Example:

.. code-block:: python

   from fs_gplib.DistributedComputing.DistributedComputing import load_partition

   sub_data, sub_targets = load_partition("./dataset/graph_parts", 0)

How to Connect a Partition to a Model
-------------------------------------

Once a partition is loaded, it can be passed directly to a static-graph model,
because the partitioned result is still a standard PyG ``Data`` object:

.. code-block:: python

   from fs_gplib.Epidemics import SIRModel

   sub_data, sub_targets = load_partition("./dataset/graph_parts", 0)

   model = SIRModel(
       sub_data,
       seeds=[0, 1, 2],
       infection_beta=0.01,
       recovery_lambda=0.005,
       device=0,
   )

The role of ``sub_targets`` is not to instantiate the model itself, but to
tell the worker which nodes it is responsible for updating or keeping after a
local propagation step.

Practical Limitations
---------------------

The partitioning interface is intentionally minimal, so there are several
things users should handle explicitly in their own advanced workflows:

- node-state synchronisation between workers;
- communication scheduling and process-group setup;
- checkpointing and fault tolerance;
- device placement and memory balancing;
- dynamic repartitioning if the workload changes over time.

For dynamic-network models, extra care is also required because users need to
manage a sequence of snapshots rather than a single static graph.  The current
partitioning helper is therefore most directly applicable to the static-graph
epidemic and opinion models.

When to Use This Interface
--------------------------

Use the partitioning interface when:

- the graph is too large for a single device;
- you need full control over multi-process or multi-machine execution;
- you want to integrate ``FS_GPlib`` propagation kernels into an existing
  distributed infrastructure.

If your graph already fits on one device, or if your main goal is simply to
run many Monte-Carlo simulations quickly, the built-in batch interfaces
described in :doc:`/tutorial` are usually the simpler and more efficient
choice.
