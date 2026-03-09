Test
====

The Independent Cascades (IC) model describes information diffusion or
contagion processes on a static graph :math:`G=(V,E)`,
where each node can be either **active (infected and recovered)** or **inactive (susceptible)**.
Once a node becomes active, it gets **one chance** to activate each of its inactive neighbors with a given probability (**infected**);
whether or not it succeeds, it has no more chance to activate others (**recovered**).

