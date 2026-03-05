API Documentation Template
==========================

This page demonstrates how to document Python APIs, including classes, functions, and mathematical formulas.

Mathematical Formulae
---------------------

Inline formula: :math:`E = mc^2`

Block formula:

.. math::

   \nabla \cdot \vec{E} = \frac{\rho}{\varepsilon_0}


Function Documentation (Manual)
-------------------------------

.. py:function:: calculate_energy(mass: float, c: float = 3e8) -> float

   Calculates energy using the formula :math:`E = mc^2`.

   :param mass: The mass of the object.
   :type mass: float

   :param c: Speed of light in vacuum. Default is 3e8.
   :type c: float

   :return: Computed energy value.
   :rtype: float


Class Documentation (Manual)
----------------------------

.. py:class:: GravNetConv(in_channels: int, out_channels: int, space_dimensions: int, propagate_dimensions: int, k: int, num_workers: Optional[int] = None, **kwargs)

   The GravNet operator from the `"Learning Representations of Irregular Particle-detector Geometry with Distance-weighted Graph Networks" <https://arxiv.org/abs/1902.07987>`_ paper.

   :param in_channels: Size of each input sample, or ``-1`` to infer from input.
   :type in_channels: int

   :param out_channels: Number of output channels.
   :type out_channels: int

   :param space_dimensions: Dimension of the projection space.
   :type space_dimensions: int

   :param propagate_dimensions: Feature dimension propagated through the graph.
   :type propagate_dimensions: int

   :param k: Number of nearest neighbors.
   :type k: int

   :param num_workers: Number of parallel workers.
   :type num_workers: Optional[int]


Autodoc Example
---------------

If you enable ``sphinx.ext.autodoc`` in ``conf.py``, you can document modules automatically:

.. automodule:: mymodule
   :members:
   :undoc-members:
   :show-inheritance: