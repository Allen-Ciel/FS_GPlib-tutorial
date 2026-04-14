Custom Model
============

``FS_GPlib`` is designed so that new propagation models can be added with the
same interface as the built-in models.  For all three model families,
customisation follows the same pattern: each ``base.py`` defines two base
classes, one for the high-level simulation API and one for the low-level
message-passing kernel.

Before implementing a new model, it is recommended to first read the
corresponding family overview:

- :doc:`/library/epidemic`
- :doc:`/library/opinion`
- :doc:`/library/dynamic_network`

Which Base File Should You Start From?
--------------------------------------

Choose the ``base.py`` file that matches your model family.

.. list-table::
   :header-rows: 1
   :widths: 18 32 25 25

   * - Family
     - Base file
     - High-level base class
     - Low-level base class
   * - Epidemic
     - ``fs_gplib/Epidemics/base.py``
     - ``DiffusionModel``
     - ``Diffusion_process``
   * - Opinion
     - ``fs_gplib/Opinions/base.py``
     - ``DiffusionModel``
     - ``Diffusion_process``
   * - Dynamic network
     - ``fs_gplib/Dynamic/base.py``
     - ``DiffusionModel``
     - ``Diffusion_process``

Although the class names are the same, the three families are not identical:

- Epidemic models are defined on static graphs and usually use discrete node
  states such as susceptible, infected, or recovered.
- Opinion models are also defined on static graphs, but may use either
  discrete or continuous node values.
- Dynamic models operate on a sequence of graph snapshots, so the process class
  must read ``edge_index_list`` and possibly ``edge_attr_list`` step by step.

Two Layers of a Custom Model
----------------------------

In a custom implementation, the two base classes play different roles.

``DiffusionModel``: simulation lifecycle and public API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The high-level model class is responsible for the user-facing interface:

- validating the input graph or snapshot sequence;
- parsing seeds or initial opinions;
- validating model parameters;
- initialising ``node_status``;
- moving data and tensors to CPU or GPU;
- exposing ``run_iteration``, ``run_iterations``, ``run_epoch``, and
  ``run_epochs``.

In practice, most custom model classes follow the same structure as the built-in
models:

1. Define ``__init__`` and pass model-specific parameters to ``super().__init__``.
2. Implement ``_init_node_status`` to build the node-state tensor(s).
3. Implement ``_set_device`` to move tensors and create the process object.
4. Implement ``run_iteration`` and ``run_iterations`` for step-by-step
   simulation.
5. Implement ``run_epoch`` and ``run_epochs`` if you want Monte-Carlo and
   batch-parallel execution.
6. Implement ``_return_final`` to map the internal state representation to the
   returned tensor.

``Diffusion_process``: propagation rule and tensor update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process class is the message-passing kernel built on top of
``torch_geometric.nn.MessagePassing``.  It is responsible for the actual update
rule at each time step:

- ``forward`` defines how the full state tensor evolves through one or multiple
  iterations;
- ``message`` defines the edge-level interaction;
- ``aggregate`` can be overridden when the default ``sum`` or ``mean`` is not
  sufficient.

For epidemic and static opinion models, the usual pattern is to repeat the
current node state to shape ``(epochs, N, 1)`` and evolve several Monte-Carlo
realisations in parallel.  For dynamic models, ``forward`` additionally needs
to read the current snapshot according to ``self.times``.

Minimal Development Checklist
-----------------------------

When adding a new custom model, it is useful to check the following questions:

- Which base family matches the model: epidemic, opinion, or dynamic?
- What is the node-state representation inside ``node_status``?
- Which parameters should be validated in ``__init__``?
- Does the model require only ``message`` and default aggregation, or also a
  custom ``aggregate``?
- Should ``run_epochs`` support batch parallelism?
- What tensor format should the final result return?

Example 1: SI with Randomly Blocked Susceptible Nodes
-----------------------------------------------------

This example extends the SI model by introducing a third state, ``Blocked``.
At each iteration, a proportion ``blocking_rate`` of currently susceptible
nodes is randomly moved to ``Blocked``.  Blocked nodes are never infected
again.

We use the following update order at each step:

1. Sample susceptible nodes that become blocked.
2. Compute infections only on the remaining susceptible nodes.

The model therefore has three final states:

- ``0``: susceptible
- ``1``: infected
- ``2``: blocked

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

    from tqdm import tqdm

    from fs_gplib.Epidemics.base import DiffusionModel, Diffusion_process
    from fs_gplib.utils import *


    class BlockedSIModel(DiffusionModel):
        def __init__(self,
                    data,
                    seeds,
                    infection_beta,
                    blocking_rate,
                    device='cpu',
                    use_weight=False,
                    rand_seed=None):
            super().__init__(
                data=data,
                seeds=seeds,
                rand_seed=rand_seed,
                device=device,
                use_weight=use_weight,
                infection_beta=infection_beta,
                blocking_rate=blocking_rate,
            )

        def _init_node_status(self):
            self.node_status = {
                "I": get_binary_mask(self.data.num_nodes, self.seeds).bool().to(self.device),
                "B": torch.zeros((self.data.num_nodes, 1), dtype=torch.bool, device=self.device),
            }

        def _set_device(self, device):
            super()._set_device(device)
            self.data = self.data.to(self.device)
            self._init_node_status()
            edge_attr = self.data.edge_attr if self.use_weight else None
            self.model = BlockedSI_process(
                self.data.edge_index,
                infection_beta=self.infection_beta,
                blocking_rate=self.blocking_rate,
                edge_attr=edge_attr,
            )

        def run_iteration(self):
            return self.run_iterations(1)

        def run_iterations(self, times):
            check_int(times=times)

            self.model._set_iterations(times)
            out_all = self.model(self.node_status)
            self.node_status["I"] = out_all["I"].squeeze(0)
            self.node_status["B"] = out_all["B"].squeeze(0)
            return self._return_final(out_all)

        def run_epoch(self, iterations_times):
            return self.run_epochs(1, iterations_times, 1)

        def run_epochs(self, epochs, iterations_times, batch_size=200):
            check_int(
                    epochs=epochs,
                    iterations_times=iterations_times,
                    batch_size=batch_size,
                )

            self._init_node_status()
            epoch_groups = epochs_groups_list(epochs, batch_size)
            bar = tqdm(epoch_groups)
            finals = []

            with torch.no_grad():
                for i, epoch_group in enumerate(bar):
                    bar.set_description(f"Batch {i}")
                    self.model._set_iterations(iterations_times)
                    out_all = self.model(self.node_status, epoch_group)
                    finals.append(self._return_final(out_all).to("cpu"))

            return torch.cat(finals, dim=0)

        def _return_final(self, out_all):
            infected = out_all["I"]
            blocked = out_all["B"]
            final = torch.zeros_like(infected, dtype=torch.long)
            final[infected] = 1
            final[blocked] = 2
            return final.squeeze(-1)


    class BlockedSI_process(Diffusion_process):
        def __init__(self, edge_index, infection_beta, blocking_rate, edge_attr=None):
            super().__init__(
                edge_index=edge_index,
                infection_beta=infection_beta,
                blocking_rate=blocking_rate,
                edge_attr=edge_attr,
            )

        def forward(self, node_status, epochs=1):
            infected = node_status["I"].unsqueeze(0).repeat(epochs, 1, 1)
            blocked = node_status["B"].unsqueeze(0).repeat(epochs, 1, 1)

            while self.iterations_times > self.times:
                susceptible = (~infected) & (~blocked)

                block_rand = torch.rand_like(infected, dtype=torch.float32)
                new_blocked = susceptible & (block_rand < self.blocking_rate)
                blocked[new_blocked] = True

                susceptible = (~infected) & (~blocked)
                temp = self.propagate(self.edge_index, x=infected.float())
                infection_prob = 1 - torch.exp(temp)
                infect_rand = torch.rand_like(infected, dtype=torch.float32)
                new_infected = susceptible & (infect_rand < infection_prob)
                infected[new_infected] = True

                self.times += 1

            return {"I": infected, "B": blocked}

        def message(self, x_j):
            return torch.log(1 - self.infection_beta * self.edge_attr * x_j)

Usage
~~~~~

.. code-block:: python

   import torch
   from torch_geometric.data import Data

   edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 3]])
   data = Data(x=torch.zeros((4, 1)), edge_index=edge_index)

   model = BlockedSIModel(
       data=data,
       seeds=[0],
       infection_beta=0.2,
       blocking_rate=0.1,
       device="cpu",
   )

   final = model.run_epochs(epochs=100, iterations_times=20, batch_size=50)



Example 2: Friedkin-Johnsen Opinion Model
-----------------------------------------

The Friedkin-Johnsen (FJ) model is a classical continuous opinion model.  Each
node :math:`i` keeps an initial opinion :math:`x_i^{(0)}` and, at every round,
balances social influence with its own prejudice:

.. math::

   x_i^{(k+1)} = \lambda x_i^{(0)} + (1-\lambda)\frac{1}{|N(i)|}\sum_{j \in N(i)} x_j^{(k)},

where :math:`\lambda \in [0,1]` is the stubbornness parameter.  A larger
:math:`\lambda` means the node stays closer to its initial opinion.

Below we implement the simplest version with:

- continuous opinions in ``[0, 1]``;
- a single global stubbornness coefficient ``lambda``;
- the average of neighbors as the interpersonal influence term.

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   import random

   import torch_scatter
   from tqdm import tqdm

   from fs_gplib.Opinions.base import DiffusionModel, Diffusion_process
   from fs_gplib.utils import *


   class FriedkinJohnsenModel(DiffusionModel):
       def __init__(self, data, seeds, stubbornness, device="cpu", rand_seed=None):
           super().__init__(
               data=data,
               seeds=seeds,
               rand_seed=rand_seed,
               device=device,
               stubbornness=stubbornness,
           )

       def _initialize_seeds(self, seeds):
           self.num_nodes = self._get_num_nodes(self.data)

           if seeds is None:
               random.seed(self.rand_seed)
               self.seeds = [random.uniform(0.0, 1.0) for _ in range(self.num_nodes)]
           elif isinstance(seeds, list):
               if len(seeds) != self.num_nodes:
                   raise ValueError("Number of seeds must equal the number of nodes.")
               if min(seeds) < 0.0 or max(seeds) > 1.0:
                   raise ValueError("All initial opinions must be in [0, 1].")
               self.seeds = seeds
           else:
               raise ValueError("seeds must be a list of floats or None.")

       def _init_node_status(self):
           initial = torch.tensor(self.seeds, dtype=torch.float32).unsqueeze(1).to(self.device)
           self.node_status = {
               "SI": initial.clone(),
               "X0": initial.clone(),
           }

       def _set_device(self, device):
           super()._set_device(device)
           self.data = self.data.to(self.device)
           self._init_node_status()
           self.model = FriedkinJohnsen_process(
               self.data.edge_index,
               stubbornness=self.stubbornness,
               iterations_times=0,
           )

       def run_iteration(self):
           return self.run_iterations(1)

       def run_iterations(self, times):
           
           check_int(times=times)

           self.model._set_iterations(times)
           out_all = self.model(self.node_status)
           self.node_status["SI"] = out_all.squeeze(0)
           return self._return_final(out_all)

       def run_epoch(self, iterations_times):
           return self.run_epochs(1, iterations_times, 1)

       def run_epochs(self, epochs, iterations_times, batch_size=200):
           check_int(
               epochs=epochs,
               iterations_times=iterations_times,
               batch_size=batch_size,
           )

           self._init_node_status()
           epoch_groups = epochs_groups_list(epochs, batch_size)
           bar = tqdm(epoch_groups)
           finals = []

           with torch.no_grad():
               for i, epoch_group in enumerate(bar):
                   bar.set_description(f"Batch {i}")
                   self.model._set_iterations(iterations_times)
                   out_all = self.model(self.node_status, epoch_group)
                   finals.append(self._return_final(out_all).to("cpu"))

           return torch.cat(finals, dim=0)

       def _return_final(self, out_all):
           return out_all.float().squeeze(-1)


   class FriedkinJohnsen_process(Diffusion_process):
       def __init__(self, edge_index, stubbornness, iterations_times):
           super().__init__(
               edge_index=edge_index,
               aggr=None,
               stubbornness=stubbornness,
               iterations_times=iterations_times,
           )

           self.edge_index, _ = add_self_loops(edge_index)

       def forward(self, node_status, epochs=1):
           x = node_status["SI"].repeat(epochs, 1, 1)
           x0 = node_status["X0"].repeat(epochs, 1, 1)

           while self.iterations_times > self.times:
               neighbor_mean = self.propagate(self.edge_index, x=x)
               updated = self.stubbornness * x0 + (1 - self.stubbornness) * neighbor_mean

               mask = ~torch.isnan(updated)
               x[mask] = updated[mask]

               self.times += 1

           return x

       def message(self, x_j):
           return x_j, torch.ones_like(x_j)

       def aggregate(self, inputs, ptr=None, dim_size=None):
           opinion_sum = torch_scatter.scatter_add(
               inputs[0], self.edge_index[1], dim=1, dim_size=dim_size
           )
           opinion_count = torch_scatter.scatter_add(
               inputs[1], self.edge_index[1], dim=1, dim_size=dim_size
           )
           return opinion_sum / opinion_count


Usage
~~~~~

.. code-block:: python

   import torch
   from torch_geometric.data import Data

   edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
   data = Data(x=torch.zeros((4, 1)), edge_index=edge_index)

   initial_opinion = [0.1, 0.4, 0.8, 0.9]

   model = FriedkinJohnsenModel(
       data=data,
       seeds=initial_opinion,
       stubbornness=0.3,
       device="cpu",
   )

   final = model.run_epochs(epochs=50, iterations_times=30, batch_size=25)

This example is a good template when you need continuous opinion values and a
custom aggregation rule.

Batch-Parallel and Distributed Use
----------------------------------

The two examples above are already batch-ready because they follow the same
pattern as the built-in models:

- ``run_epochs`` splits ``epochs`` into groups with ``epochs_groups_list``;
- the process class repeats the current state with ``repeat(epochs, 1, 1)``;
- each batch returns a tensor of shape ``(epochs, N)`` after squeezing the last
  dimension.

For distributed execution, ``FS_GPlib`` currently provides graph partitioning
utilities but does not hide the full distributed training or simulation loop.
Therefore, a custom model can be integrated into the workflow described in
:doc:`/tutorial` as long as it keeps the same public interface and exposes
``node_status`` in a form that can be synchronised across processes.

Practical Suggestions
---------------------

- If your model is a small variation of an existing implementation, start by
  copying the closest built-in model rather than writing from scratch.
- If your update rule only changes edge interaction, you often only need to
  modify the process class.
- If your model introduces new compartments or opinion tensors, first decide
  how they should be stored in ``node_status``, then implement
  ``_return_final``.
- For dynamic networks, it is usually easiest to start from the closest model
  in ``fs_gplib/Dynamic`` and adapt its ``forward`` method to the new rule.
