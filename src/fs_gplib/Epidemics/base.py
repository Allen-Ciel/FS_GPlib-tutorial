import sys
import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
import warnings

from ..utils import *

class DiffusionModel:
    def __init__(self,
                 data,
                 seeds: Union[float, List[int], None],
                 rand_seed=None,
                 device='cpu',
                 use_weight=False,
                 **kwargs):
        '''
        :param data: PyG Data
        :param seeds: if float, the percentage of seeds
        :param rand_seed: random seed
        '''

        np.random.seed(rand_seed)

        if use_weight not in [True, False]:
            raise ValueError("Parameter 'use_weight' must be either True or False.")
        self.use_weight = use_weight

        self._validate_graph(data)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)

    def _validate_graph(self, data):
        if not isinstance(data, Data):
            raise ValueError("data must be a Data object from the PyG library.")
        if data.edge_index == None:
            raise ValueError("data must contain edge_index.")
        if self.use_weight:
            if data.edge_attr == None:
                raise ValueError("data does not have edge weights.")
        if data.x == None:
            if data.num_nodes == None:
                num = data.edge_index.max().item()+1
                data.x = torch.zeros((num,1), dtype=torch.long)
            elif isinstance(data.num_nodes, int):
                data.x = torch.zeros((data.num_nodes, 1), dtype=torch.long)

        self.data = data

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
            if not (0 < seeds < 1):
                raise ValueError("When seeds are decimal numbers, they must be in the range (0,1).")
            seed_count = int(self.num_nodes * seeds)
            random_seeds_list = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
            self.seeds = random_seeds_list#torch.tensor(random_seeds_list)
        elif isinstance(seeds, list):
            if not all(isinstance(i, int) for i in seeds):
                raise ValueError("When seeds are a list, the elements must be integers.")
            if max(seeds) > self.num_nodes:
                seeds = torch.tensor(seeds)
                valid_seeds = seeds[seeds<self.num_nodes]
                removed = seeds.shape[1] - valid_seeds.shape[1]
                if removed > 0:
                    warnings.warn(f"Removed {removed} out_of_range seed index. Valid seed indices are 0 to {self.num_nodes-1}.", UserWarning)
                seeds = valid_seeds.tolist()

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
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps:{}'.format(device))
            else:
                raise Exception("No supported GPU device (MPS or CUDA) is available.")
    def _validate_parameters(self, kwargs):
        try:
            check_float_parameter(0, 1, True, True, **kwargs)
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
        '''
        :param iterations_times: timestep
        :param epochs: MC times
        '''
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


    def run_iteration(self):
        pass

    def _return_final(self):
        pass

class Diffusion_process(MessagePassing):
    def __init__(self, edge_index, aggr='add', **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)
        self.edge_attr = None
        self.edge_index = edge_index
        self.device = self.edge_index.device
        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.unsqueeze(0).unsqueeze(2)
        else:
            self.edge_attr = torch.tensor([1]).to(
                self.device)

    def _set_iterations(self, iterations_times):
        self.iterations_times = iterations_times
        self.times = 0