Tutorial
============

Example
-------

Import
~~~~~~

导入需要的包，并选择传播模型

.. code-block:: python

   import torch
   from torch_geometric.datasets import Planetoid
   from torch_geometric.utils import degree
   from FS_GPlib.Epidemics.SIRModel import SIRModel

Data
~~~~

读取图数据，使用PyG包的数据，其数据结构为压缩矩阵行。

.. code-block:: python

    dataset = Planetoid(root='./dataset', name='Cora')
    data = dataset[0]


若数据为不是压缩矩阵行结构，需要进行转换。数据结构构建见 `Data <https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.Data.html#torch_geometric.data.Data>`_。

若数据是Networkx.Graph()，可以 ``from_networkx`` 进行转换

.. code-block:: python

    import networkx as nx
    from torch_geometric.utils import from_networkx

    G = nx.florentine_families_graph()
    data = from_networkx(G)

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))

.. note::
    注意两个数据格式对节点的编码方式不一样，需要通过mapping进行对照。

Seeds
~~~~~

种子节点的格式可以是 ``list`` 或者 ``float`` 。

种子集合可以是通过各种算法计算或者用户指定的节点列表，以下是一个简单示例，通过度中心性选取前10%的节点作为种子：

.. code-block:: python

    def Degree_centrality_seeds(data, num_seeds):
        target = data.edge_index[1]
        N = data.num_nodes
        D = degree(target, N)
        _, index = torch.sort(D, descending=True)
        seeds = index[:num_seeds]
        return seeds.tolist()

    num_seeds = int(data.num_nodes*0.1)
    seeds = Degree_centrality_seeds(data, num_seeds)

当然也可以通过随机生成的方式生成种子集合，细节看下一节Model。

Instantiate Model
~~~~~~~~~~~~~~~~~

导入数据，种子集合，以及确认模型参数之后构建模型：

.. code-block:: python

    i_beta=0.01 # infection
    r_lambda=0.005 # recovery
    device = 1 # device
    model = SIRModel(data, seeds, i_beta, r_lambda, device)

每个模型所需的参数见模型介绍部分。

如果没有种子集合，希望通过随机方式生成种子集合，将seeds设置为 ``float`` ，可以生成对应比例的种子集合：

.. code-block:: python

    seeds = 0.3 # ratio of seeds
    i_beta=0.01 # infection
    r_lambda=0.005 # recovery
    device = 1 # device
    model = SIRModel(data, seeds, i_beta, r_lambda, device)

    # check the seeds set
    print(model.seeds)

如果希望随机生成的种子可以复现，请在构建模型时加入随机种子：

.. code-block:: python

    seeds = 0.3 # ratio of seeds
    i_beta=0.01 # infection
    r_lambda=0.005 # recovery
    device = 1 # device
    model = SIRModel(data, seeds, i_beta, r_lambda, device, rand_seed=33)

    # check the seeds set
    print(model.seeds)

Execute simulation
~~~~~~~~~~~~~~~~~~

可以选择四种不同的运行接口：

- ``Model.run_iteration()``: Execute one time step from the current node state and return the updated node state.
- ``Model.run_iterations(times)``: Execute multiple time steps starting from the current node state and return the final node state after all iterations.
- ``Model.run_epoch(times)``: Execute multiple time steps starting from the initial state and return the final node state after all iterations.
- ``Model.run_epochs(epochs, times, batch_size)``: Perform multiple Monte Carlo simulations in batches, each starting from the initial state, and return the final node states.

以下是一个运行示例：

.. code-block:: python

    MC = 10000 # simulation times
    it = 100 # iteration times
    bs = 2000 # batch size
    finals = model.run_epochs(MC, it, bs)

Example code
------------

一份完整的示例：

.. code-block:: python

    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import degree
    from FS_GPlib.Epidemics.SIRModel import SIRModel

    def Degree_centrality_seeds(data, num_seeds):
        target = data.edge_index[1]
        N = data.num_nodes
        D = degree(target, N)
        _, index = torch.sort(D, descending=True)
        seeds = index[:num_seeds]
        return seeds.tolist()

    dataset = Planetoid(root='./dataset', name='Cora')
    data = dataset[0]
    num_seeds = int(data.num_nodes*0.1)
    seeds = Degree_centrality_seeds(data, num_seeds)

    i_beta=0.01
    r_lambda=0.005
    device = 1
    model = SIRModel(data, seeds, i_beta, r_lambda, device)

    MC = 10000
    it = 100
    bs = 2000
    finals = model.run_epochs(MC, it, bs)

    count = (finals>0).sum().item()/MC
    print(f'Final spread range is {count}')

An example for DySIRModel:

.. code-block:: python

    import torch
    import random
    from torch_geometric.datasets import BitcoinOTC
    import matplotlib.pyplot as plt
    import numpy as np

    from fs_gplib.Dynamic.DySIRModel import DySIRModel

    def plot_dy_sir(finals):
        # finals.shape: (num_timesteps (T), MC, num_nodes)
        if isinstance(finals, torch.Tensor):
            finals_np = finals.cpu().numpy()
        else:
            finals_np = np.array(finals)

        T = finals_np.shape[0]
        MC = finals_np.shape[1]
        N = finals_np.shape[2]

        all_states = np.unique(finals_np)
        avg_state_cnt = []
        for t in range(T):
            state_cnt = []
            for s in all_states:
                # (MC, N) == s -> sum over nodes, then mean over MC runs
                count = np.sum(finals_np[t] == s, axis=1)
                state_cnt.append(np.mean(count))
            avg_state_cnt.append(state_cnt)
        avg_state_cnt = np.array(avg_state_cnt)  # (T, num_state)

        for idx, s in enumerate(all_states):
            plt.plot(range(T), avg_state_cnt[:, idx], label=f"State {s}")

        plt.xlabel("Time Step")
        plt.ylabel("Average #Nodes")
        plt.title("Average Number of Nodes per State Over Time")
        plt.legend(["S", "I", "R"])
        plt.tight_layout()
        plt.show()



    dataset = BitcoinOTC(root="./dataset")

    data = dataset[0]
    seeds = data.edge_index[1].unique().tolist() 

    # use weight or not
    use_weight = False
    if use_weight:
        print("Using weight")
        random.seed(42)
        edge_attr = torch.tensor(
            [random.random() for _ in range(data.edge_index.shape[1])]
        )
        data.edge_attr = edge_attr

    infection_beta = 0.1
    recovery_lambda = 0.05
    device = 'cpu' #0
    MC = 1000

    x = torch.arange(data.num_nodes)
    edge_index_list = [dataset[i].edge_index for i in range(len(dataset))]

    model = DySIRModel(x, edge_index_list, seeds, infection_beta, recovery_lambda, device, use_weight)
    finals = model.run_epochs(MC, batch_size=100)

    plot_dy_sir(finals)


The Output:

.. image:: ./images/result_for_DySIRModel.png
   :alt: result for DySIRModel
   :align: center
   :width: 70%
    
About Batch
-----------

Batch并行算法宏观加速的核心，用于 ``Model.run_epochs(epochs, times, batch_size)`` 。

How to use batch
~~~~~~~~~~~~~~~~

Batch并行使得多个模拟可以同时进行，但batch大小不能无限放大，而是受硬件性能限制。batch大小设置过大时会提示out of memory，通过实验验证并不是batch 大小越大运行效率越高。

用户可以通过指数级递减的方式寻找合适的batch大小。

About Distributed
-----------------

分布式方法适用于单机无法处理的超大规模数据。分布式方法的核心是按目标节点划分数据，实现子图之间数据量不倾斜，并且每个时间步只进行一次同步。

该方法可以在单机单卡设备、单机多卡设备和多机多卡设备上实现超大规模图的传播模拟。

How to use distributed computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

算法库不集成分布式代码，但提供了数据划分方法。

并给出一个实现案例，`test_DC.py <coming soon.>`_

该代码运行方式为：

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nnodes=1 --nproc_per_node=6 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 test_DC.py -d Webbase -s 100 -mc 1000

命令行解读：

- ``CUDA_VISIBLE_DEVICES=1,2,3,4,5,6``

    Restricts the script to only use GPUs with device IDs 1 through 6. These are the GPU IDs made visible to the process.

- ``torchrun``

    The PyTorch launcher used for distributed training/simulation across multiple processes or GPUs.

- ``--nnodes=1``

    Specifies that only one physical machine (node) is involved in the computation.

- ``--nproc_per_node=6``

    Launches 6 separate processes on the single node, typically one per GPU (corresponding to the 6 visible GPUs).

- ``--node_rank=0``

    Indicates the rank of this node in a multi-node setup. Since there’s only one node, it’s rank 0.

- ``--master_addr=127.0.0.1``

    Sets the IP address of the master node (localhost in this case).

- ``--master_port=12345``

    Sets the port used for inter-process communication.

- ``test_DC.py``

    The Python script to be executed, which performs a distributed SIR simulation.

- ``-d Webbase``

    Specifies the dataset to use (Webbase in this case).

- ``-s 100``

    Sets the number of simulation steps per Monte Carlo run to 100.

- ``-mc 1000``

    Specifies that 1000 Monte Carlo simulations will be run.