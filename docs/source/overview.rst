Overview
========

**FS_GPlib** is a flexible and scalable Python library designed for simulating a wide range of graph-based propagation processes. It supports classical epidemic models, opinion dynamics, and simulations on dynamic networks. The library is built to facilitate high-efficiency simulation through support for Message Passing based dual-acceleration framework and distributed computing.

This documentation provides a structured overview of FS_GPlib, from installation and usage to the implementation of custom models. Whether you're a researcher studying network diffusion or a developer looking to integrate propagation models into your pipeline, FS_GPlib offers a modular and extensible foundation.

Motivation
----------

Propagation models are essential tools in understanding how information, behaviors, or diseases spread through complex networks. FS_GPlib is developed to address the following challenges:

- The need for **efficient simulation** on large-scale graphs.
- The demand for **modular design** that supports custom models.
- The importance of **distributed simulation** to overcome scalability bottlenecks.

Core Capabilities
-----------------

- **Model Support**: Built-in support for classical epidemic models (SI, SIS, SIR), opinion dynamics, and network evolution processes.
- **Dual-acceleration**: Support for Message Passing based dual-acceleration framework.
- **Distributed Execution**: Supports graph partitioning to enable distribution across multiple processors or nodes.
- **Extensibility**: Easily implement your own propagation rules through the custom model interface.

Project Structure
-----------------
.. note::



The documentation is organized into the following sections:
    - **Installation**: Set up the required environment and dependencies.
    - **Tutorial**: Learn the basics through example-driven instruction.
    - **Library**: Explore built-in models, distributed options, and configurable parameters.
    - **Custom Model**: Create your own propagation logic by extending the framework.
    - **Advanced**: Explore advanced features for Ultra-Large Graph Propagation.
    
Get Started
-----------

To begin using FS_GPlib, proceed to the :doc:`installation` section and follow the setup instructions. Then explore the :doc:`tutorial` for example workflows and usage patterns.