Installation
============

Environment
-----------

This section describes how to prepare the environment necessary for running FS_GPlib.

Python Version
~~~~~~~~~~~~~~

FS_GPlib is developed and tested with:

- Python 3.10

We recommend using a virtual environment to avoid conflicts with existing packages.

Create and activate a virtual environment:

.. code-block:: bash

   conda create -n fs_gplib_env python=3.10
   conda activate fs_gplib_env



Install
-------

Install from TestPyPI
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fs-gplib

Required Packages
~~~~~~~~~~~~~~~~~

If you encounter environment errors, try installing (or pinning) the following dependencies first, and then install ``fs-gplib``:

- ``python==3.10``
- ``torch==2.1.2``
- ``torch_geometric==2.5.3``
- ``numpy==1.24.1``
- ``tqdm==4.64.1``
- ``scipy==1.10.0``

FS_GPlib requires PyTorch and PyTorch Geometric (PyG).
``torch_scatter`` is additionally required when you use models that depend on scatter operators
(currently ``HKModel`` and ``WHKModel`` in ``fs_gplib.Opinions``).
You can install PyTorch/PyG by following the official instructions:

- PyTorch: https://pytorch.org/get-started/locally/
- PyG: https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html

If you need ``torch_scatter``, install the wheel that matches your PyTorch and CUDA/CPU build.
For compatibility details and wheel links, see:

- https://data.pyg.org/whl/

Example (CPU; change version tag to match your environment):

.. code-block:: bash

   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

Test Installation
~~~~~~~~~~~~~~~~~

To verify the installation was successful, try importing the library in Python:

.. code-block:: python

   import fs_gplib
   print(fs_gplib.__version__)

If no errors are raised, the installation is complete.
