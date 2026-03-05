Installation
============

Environment
-----------

This section describes how to prepare the environment necessary for running FS_GPlib.

Python Version
~~~~~~~~~~~~~~


FS_GPlib is developed and tested with:

- Python 3.8 or above

We recommend using a virtual environment to avoid conflicts with existing packages.

Create and activate a virtual environment:

.. code-block:: bash

   python3 -m venv fs_gplib_env
   source fs_gplib_env/bin/activate   # On Windows use: fs_gplib_env\Scripts\activate

Required Packages
~~~~~~~~~~~~~~~~~

The core dependencies include:

- ``numpy`` and ``scipy`` for numerical operations
- ``networkx`` for graph construction
- ``torch`` for tensor-based batch simulation
- ``mpi4py`` (optional) for distributed simulations

You can install the dependencies manually, or use the provided `requirements.txt`:

.. code-block:: bash

   pip install -r requirements.txt

GPU Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~

If you plan to run FS_GPlib on a GPU-enabled system, ensure that:

- CUDA and cuDNN are properly installed
- You install the appropriate version of PyTorch with GPU support:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

You can verify PyTorch sees the GPU:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())  # Should print: True



Install
-------

This guide explains how to install FS_GPlib from source or via cloning the repository.

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

First, clone the FS_GPlib project from GitHub:

.. code-block:: bash

   git clone None
   cd FS_GPlib

Then install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Install in Editable Mode (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to make modifications to the source code and have them reflected without reinstalling, use editable mode:

.. code-block:: bash

   pip install -e .

Test Installation
~~~~~~~~~~~~~~~~~

To verify the installation was successful, you can try importing the library in Python:

.. code-block:: python

   import fs_gplib
   print(fs_gplib.__version__)

If no errors are raised, the installation is complete.
