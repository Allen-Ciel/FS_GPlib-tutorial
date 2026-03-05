import sys
import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from ..utils import *

class DiffusionModel:
    def __init__(self, data, seeds: Union[float, List[int], None], rand_seed=None, device='cpu', **kwargs):
        '''
        :param data: PyG Data
        :param seeds: if float, the percentage of seeds
        :param rand_seed: random seed
        '''
        self._skip_init = False
        self.rand_seed = rand_seed

        self._validate_graph(data)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)

    def _validate_graph(self, data):

        if not isinstance(data, Data):
            raise ValueError("data must be a Data object from the PyG library.")
        self.data = data

        self.is_directed = self.data.is_directed()

    def _get_num_nodes(self, data):
        if hasattr(data, 'num_nodes'):
            return data.num_nodes
        elif hasattr(data, 'x') and isinstance(data.x, torch.Tensor):
            return data.x.size(0)
        else:
            raise ValueError("The number of nodes in data cannot be determined.")

    def _initialize_seeds(self, seeds):
        """
        Initialize self.seeds based on input seeds. If it's a float, convert it
        to a list of randomly selected nodes.
        """

        self.num_nodes = self._get_num_nodes(self.data)

        if isinstance(seeds, (float, int)):
            if not (0 <= seeds < 1):
                raise ValueError("When seeds are decimal numbers, they must be in the range (0,1).")
            # 按比例随机生成种子节点列表
            np.random.seed(self.rand_seed)  # 用于按照比例随机生成种子
            seed_count = int(self.num_nodes * seeds)
            self.seeds = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
        elif isinstance(seeds, list):
            if not all(isinstance(i, int) for i in seeds):
                raise ValueError("When seeds are a list, the elements must be integers.")
                # 检查最大值是否超出 data 的节点个数
            if max(seeds) >= self._get_num_nodes(self.data):
                raise ValueError("The maximum value in the seeds list cannot exceed the number of nodes in data.")
            self.seeds = seeds
        elif seeds is None:
            self.seeds = seeds
        else:
            raise ValueError("seeds must be decimals in the range (0,1), a list of integers, or None.")

    def _set_device(self, device):
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:{}'.format(device))
            else:
                raise Exception(f"cuda is not available!")

    def _validate_parameters(self, kwargs):

        try:
            check_parameter(**kwargs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _init_node_status(self):
        pass

    def run_epoch(self, **kwargs):
        pass

    def run_epochs(self, epochs, iterations_times):

        try:
            check_int(iterations_times=iterations_times, epochs=epochs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self._init_node_status()

        bar = tqdm(range(epochs))
        final = []
        self._skip_init = True
        with torch.no_grad():
            for i in bar:
                bar.set_description('Batch {}'.format(i))
                out = self.run_epoch(iterations_times=iterations_times)
                final.append(out)
        return final


    def iteration(self):
        pass

    def _return_final(self):
        pass

class Diffusion_process(MessagePassing):
    def __init__(self, edge_index, aggr='sum', **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)
        self.edge_index = edge_index
        self.device = self.edge_index.device
        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _set_iterations(self, iterations_times):
        self.iterations_times = iterations_times
        self.times = 0

