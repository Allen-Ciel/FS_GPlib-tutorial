import sys
import numpy as np
from typing import Union, List
from torch_geometric.nn import MessagePassing
from tqdm import tqdm


from ..utils import *

class DiffusionModel:
    def __init__(self,
                 x,
                 edge_index_list,
                 seeds: Union[float, List[int], None],
                 rand_seed=None,
                 device='cpu',
                 edge_attr_list=None,
                 **kwargs):
        '''
        :param seeds: if float, the percentage of seeds
        :param rand_seed: random seed
        '''
        np.random.seed(rand_seed)

        self.use_weight = True
        if edge_attr_list is None:
            self.use_weight = False

        self.x = x
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list

        np.random.seed(rand_seed)
        self._initialize_seeds(seeds)
        self._validate_parameters(kwargs)
        self._set_device(device)



    def _validate_graph(self, x):
        pass

    def _get_num_nodes(self):
        return self.x.shape[0]

    def _initialize_seeds(self, seeds):
        """
        Initialize self.seeds based on input seeds. If it's a float, convert it
        to a list of randomly selected nodes.
        """

        self.num_nodes = self.x.shape[0]

        if isinstance(seeds, (float, int)):
            if not (0 <= seeds < 1):
                raise ValueError("When seeds are decimal numbers, they must be in the range (0,1).")
            seed_count = int(self.num_nodes * seeds)
            self.seeds = np.random.choice(range(self.num_nodes), seed_count, replace=False).tolist()
        elif isinstance(seeds, list):
            if not all(isinstance(i, int) for i in seeds):
                raise ValueError("When seeds are a list, the elements must be integers.")
            if max(seeds) >= self.num_nodes:
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

        try:
            check_int(iterations_times=iterations_times, epochs=epochs)
        except ValueError as e:
            print("Caught error:", e)
            sys.exit(1)

        self._init_node_status()
        # pdb.set_trace()
        bar = tqdm(range(epochs))
        final = []
        self._skip_init = True
        with torch.no_grad():
            for i in bar:
                bar.set_description('第{}轮'.format(i))
                out = self.run_epoch(iterations_times=iterations_times)
                final.append(out)
        return final


    def iteration(self):
        pass

    def _return_final(self):
        pass
class Diffusion_process(MessagePassing):
    def __init__(self,
                 edge_index_list,
                 edge_attr_list,
                 aggr='add',
                 **kwargs):
        super(Diffusion_process, self).__init__(aggr=aggr)

        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list

        self.times = 0

        for param_name, value in kwargs.items():
            self.__setattr__(param_name, value)

    def _set_iterations(self):

        self.times = 0