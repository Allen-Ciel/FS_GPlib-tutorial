# FS_GPlib

<div align="center">
<img src="docs/source/_static/logo_v4.png" alt="FS_GPlib logo" width="180" />
<h1 align="center">FS_GPlib</h1>
<h3 align="center">Faster and more
Scalable python library for Graph Propagation models </h3>

[![TestPyPI](https://img.shields.io/badge/TestPyPI-available-orange)](https://test.pypi.org/project/fs-gplib/)
[![Python](https://img.shields.io/badge/python-3.10-blue)](#installation)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Docs](https://readthedocs.org/projects/fs-gplib-tutorial/badge/?version=latest)](https://fs-gplib-tutorial.readthedocs.io/en/latest/)

FS_GPlib is a fast and scalable Python library for simulating graph propagation processes on complex networks, covering classical diffusion and epidemic models such as Independent Cascade (IC), Linear Threshold (LT), Susceptible-Infected (SI), Susceptible-Infected-Susceptible (SIS), and Susceptible-Infected-Recovered (SIR), as well as opinion dynamics and dynamic-network diffusion.
It supports extensible data pipelines and custom algorithm development, and integrates greedy-strategy algorithms for influence maximization tasks to facilitate both research prototyping and large-scale experimentation.
</div>


## Documentation

Online documentation: https://fs-gplib-tutorial.readthedocs.io/en/latest/

## Installation

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fs-gplib
```

If you get any environment error, try installing the following dependencies manually and then installing `fs-gplib`.

`PyTorch` and `PyG` are required. `torch_scatter` is additionally required when you use models that depend on scatter operators (currently `HKModel` and `WHKModel` in `fs_gplib.Opinions`).

> python==3.10\
> torch==2.1.2\
> torch_geometric==2.5.3\
> numpy==1.24.1\
> tqdm==4.64.1\
> scipy==1.10.0

Install PyTorch and PyG from:
- [PyTorch install guide](https://pytorch.org/get-started/locally/)
- [PyG install guide](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html)

If you need `HKModel`/`WHKModel`, install `torch-scatter` with a wheel matching your PyTorch and CUDA/CPU build.
For wheel index and compatibility mapping, see:
- [PyG wheel index](https://data.pyg.org/whl/)

Example (CPU, change version tag to your environment):

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

## Examples

We give examples for the SIR model.

### Usage


```python 
"""
Test SIRModel after installing fs-gplib
"""

import torch
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

from fs_gplib.Epidemics import SIRModel


def degree_centrality_seeds(data, num_seeds: int):
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    deg, index = torch.sort(deg, descending=True)
    seeds = index[:num_seeds]
    return seeds.tolist()


def main():
   
    dataset_name = "Cora" # or "PubMed"

    if dataset_name == "Cora":
        dataset = Planetoid(root="./dataset", name="Cora")
    elif dataset_name == "PubMed":
        dataset = Planetoid(root="./dataset", name="PubMed")


    data = dataset[0]
    seeds = degree_centrality_seeds(data, int(data.num_nodes * 0.1))

    # use weight or not
    use_weight = False
    if use_weight:
        print("Using weight")
        random.seed(42)
        edge_attr = torch.tensor(
            [random.random() for _ in range(data.edge_index.shape[1])]
        )
        data.edge_attr = edge_attr

    infection_beta = 0.01
    recovery_lambda = 0.005
    device = 0          # or 'cpu'
    MC = 1000
    iteration_times = 100

    model = SIRModel(data, seeds, infection_beta, recovery_lambda, device, use_weight)
    finals = model.run_epochs(MC, iteration_times, batch_size=1)

    count = (finals == 1).sum().item() / MC
    print(f"Final spread range of SIR on {dataset_name} is {count}")


if __name__ == "__main__":
    main()
```

## Demo for Distributed Computing
###  `SIR_DC.py`
```python
import argparse
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid
import os
import time
from torch_geometric.data.data import DataEdgeAttr


from fs_gplib.Epidemics.SIRModel import SIRModel
from fs_gplib.DistributedComputing import (
    GraphPartitioner,
    load_partition,
)
from fs_gplib.utils import *

parser = argparse.ArgumentParser(description="Distributed SIR Simulation")
parser.add_argument("-d", "--dataset", default="Cora",
                    choices=['Cora', 'PubMed'])
parser.add_argument("-p", "--parts", type=int, default=4,
                    help="Number of partitions")
parser.add_argument("-s", "--steps", type=int, default=100,
                    help="Number of iterations")
parser.add_argument("-mc", "--MC", type=int, default=1000,
                    help="Number of MC")
parser.add_argument("-rp", "--read_pt", action="store_true", help="Read .pt files instead of re-partitioning")
args = parser.parse_args()

def get_degree_seeds(data: Data, num_seeds: int):
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    deg, idx = torch.sort(deg, descending=True)
    return idx[:num_seeds].tolist()

def load_data(dataset_name):
    if dataset_name == 'Cora':
        dataset = Planetoid(root='./dataset', name='Cora')
    elif dataset_name == 'PubMed':
        dataset = Planetoid(root='./dataset', name='PubMed')
    data = dataset[0]
    return data

if __name__ == '__main__':
    infection_beta = 0.01
    recovery_lambda = 0.005
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    world_size = torch.cuda.device_count()
    if rank == 0:
        final_count = []
        data = load_data(args.dataset)
        print(data.edge_index.device)
        seeds = get_degree_seeds(data, int(data.num_nodes * 0.1))
        print(args.read_pt)
        if args.read_pt == False:
            print('start partitioning...')
            t_s = time.time()
            partitioner = GraphPartitioner(data, world_size, './dataset/{}'.format(args.dataset))
            partitioner.generate_partition()
            print('partitioning done in {} s'.format(time.time() - t_s))
    else:
        seeds = None
    dist.barrier()

    seed_list = [seeds] if rank == 0 else [None]
    dist.broadcast_object_list(seed_list, src=0)
    seeds = seed_list[0]

    torch.serialization.add_safe_globals([DataEdgeAttr])
    sub_data, sub_targets = load_partition('./dataset/{}'.format(args.dataset), rank)

    model = SIRModel(sub_data, seeds=seeds, infection_beta=infection_beta, recovery_lambda=recovery_lambda, device=device_id)

    mask = torch.ones_like(model.node_status['SI'], dtype=torch.bool)
    mask[sub_targets] = False
    bar = tqdm(range(args.MC))

    for i, _ in enumerate(bar):
        model._init_node_status()
        bar.set_description('Batch {}'.format(i))
        for step in range(args.steps):
            model.run_iteration()

            # Synchronize node_status across all processes
            synced_status = {}
            for key, value in model.node_status.items():
                value[mask] = False
                # Convert to int for communication
                int_status = value.to(dtype=torch.uint8)

                # All-reduce across processes
                dist.all_reduce(int_status, op=dist.ReduceOp.SUM)
                # Convert back to boolean (presence across any process)
                synced_status[key] = int_status.bool()#.clone()


            model.node_status = synced_status

        if rank == 0:
            # Final result output (optional)
            final_state = model._return_final(model.node_status.values())
            infected_count = (final_state == 1).sum().item()

            final_count.append(infected_count)

    if rank == 0:

        print(f"[Rank {rank}] Final infected count: {np.mean(final_count)}")
        print(f'max: {np.max(final_count)}')
        print(f'min: {np.min(final_count)}')

    dist.barrier()
    dist.destroy_process_group()
```

### Usage
> CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 test_DC.py -d Cora -s 100 -mc 1000


- CUDA_VISIBLE_DEVICES=1,2,3

    Restricts the script to only use GPUs with device IDs 1 through 3. These are the GPU IDs made visible to the process.
- torchrun

    The PyTorch launcher used for distributed training/simulation across multiple processes or GPUs.
- --nnodes=1

   Specifies that only one physical machine (node) is involved in the computation.
- --nproc_per_node=3

   Launches 3 separate processes on the single node, typically one per GPU (corresponding to the 3 visible GPUs).
- --node_rank=0

   Indicates the rank of this node in a multi-node setup. Since there’s only one node, it’s rank 0.
- --master_addr=127.0.0.1

   Sets the IP address of the master node (localhost in this case).
- --master_port=12345

   Sets the port used for inter-process communication.
- test_DC.py

   The Python script to be executed, which performs a distributed SIR simulation.
- -d Webbase

   Specifies the dataset to use (Webbase in this case).
- -s 100

   Sets the number of simulation steps per Monte Carlo run to 100.
- -mc 1000

   Specifies that 1000 Monte Carlo simulations will be run.


## IM


```python
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from src.fs_gplib.Epidemics import SIRModel
from src.fs_gplib.InfluenceMaximization import CELFIM


def nx_to_pyg_data_with_mapping(graph: nx.Graph) -> tuple[Data, dict, dict]:
    """Convert a NetworkX graph to PyG data with bidirectional mappings."""
    original_nodes = list(graph.nodes())
    node_mapping = {node_id: idx for idx, node_id in enumerate(original_nodes)}
    reverse_mapping = {idx: node_id for node_id, idx in node_mapping.items()}

    relabeled_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    data = from_networkx(relabeled_graph)
    data.num_nodes = relabeled_graph.number_of_nodes()
    return data, node_mapping, reverse_mapping


def run_im_with_api(
    data: Data,
    node_mapping: dict,
    reverse_mapping: dict,
    seed_size: int = 5,
):
    """Run influence maximization via the fs_gplib API (CELFIM + SIRModel)."""

    model = SIRModel(
        data=data,
        seeds=None,
        infection_beta=0.01,
        recovery_lambda=0.005,
    )

    im = CELFIM(
        model=model,
        seed_size=seed_size,
        influenced_type=[1, 2],  # In SIR, both infected and recovered count as influenced.
        MC=1000,
        iterations_times=50,
        verbose=True,
    )

    selected_seeds = im.fit()
    original_id_seeds = [reverse_mapping[s] for s in selected_seeds]
    print(f"selected seeds (reindexed): {selected_seeds}")
    print(f"selected seeds (original ids): {original_id_seeds}")
    print(f"node mapping size: {len(node_mapping)}")
    return selected_seeds


if __name__ == "__main__":
    graph = nx.les_miserables_graph()
    data, node_mapping, reverse_mapping = nx_to_pyg_data_with_mapping(graph)
    seeds = run_im_with_api(
        data=data,
        node_mapping=node_mapping,
        reverse_mapping=reverse_mapping,
        seed_size=5,
    )
    print(f"seed count = {len(seeds)}")



```