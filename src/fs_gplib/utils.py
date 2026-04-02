import pickle
import torch
import gzip
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, add_self_loops

def check_parameter(**kwargs):
    # 判断参数是否在 [0, 1) 区间内
    for param_name, value in kwargs.items():
        if not (0 <= value <= 1):
            raise ValueError(f"Parameter '{param_name}' must be between 0 and 1, exclusive!")

def check_float_parameter(min_value, max_value, is_min_inclusive=True, is_max_inclusive=True, **kwargs):
    '''
    :param min_value: 范围的最小值
    :param max_value: 范围的最大值
    :param is_min_inclusive: 最小值是否包含在内（默认为闭区间）
    :param is_max_inclusive: 最大值是否包含在内（默认为闭区间）
    '''

    for param_name, value in kwargs.items():
        # 检查最小值
        if is_min_inclusive:
            if value < min_value:
                raise ValueError(f"{param_name} must be greater than or equal to {min_value}, but got {value}.")
        else:
            if value <= min_value:
                raise ValueError(f"{param_name} must be greater than {min_value}, but got {value}.")

        # 检查最大值
        if is_max_inclusive:
            if value > max_value:
                raise ValueError(f"{param_name} must be less than or equal to {max_value}, but got {value}.")
        else:
            if value >= max_value:
                raise ValueError(f"{param_name} must be less than {max_value}, but got {value}.")


def check_int(**kwargs):
    for param_name, value in kwargs.items():
        if not isinstance(value, int):
            raise ValueError(f"Parameter '{param_name}' must be an integer!")

def check_int_list(**kwargs):
    for param_name, value in kwargs.items():
        if not isinstance(value, list):
            raise ValueError(f"Parameter '{param_name}' must be a list!")
        elif isinstance(value, list):
            if not all(isinstance(i, int) for i in value):
                raise ValueError(f"Parameter '{param_name}' must be a list of Integers!")

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, 1)
    mask[indices] = 1
    return mask

def epochs_groups_list(epochs, batch_size):
    n = int(epochs / batch_size)
    epoch_groups = [batch_size] * n
    last_epoch_group = epochs - n * batch_size
    if last_epoch_group != 0:
        epoch_groups.append(last_epoch_group)
    return epoch_groups

def check_float_list(**kwargs):
    for param_name, value in kwargs.items():
        if not isinstance(value, list):
            raise ValueError(f"Parameter '{param_name}' must be a list!")
        elif isinstance(value, list):
            if all(isinstance(i, float) for i in value):
                if max(value) >= 1.0 or min(value) <= 0.0:
                    raise ValueError(f"Parameter '{param_name}' must be between 0 and 1!")

def Pokec(root=None, save=False):
    save_path = root + "/pokec.pkl"
    if save == True:
        file_node = root + "/soc-pokec-profiles.txt.gz"
        nodes_list = []
        with gzip.open(file_node, "rt") as file:
            for line in file:
                nodes = int(line.strip().split()[0]) - 1
                nodes_list.append(nodes)
        if sorted(nodes_list) == list(range(len(nodes_list))):
            x = torch.zeros((len(nodes_list),1))
        else:
            raise ValueError('Wrong Nodes List!')
        file_edge = root + "/soc-pokec-relationships.txt.gz"
        edges_list = [[], []]
        with gzip.open(file_edge, "rt") as file:
            for line in file:
                u, v = line.strip().split()
                edges_list[0].append(int(u) - 1)
                edges_list[1].append(int(v) - 1)
        edge_index = torch.tensor(edges_list)
        data = Data(x=x, edge_index=edge_index)

        with open(save_path, "wb") as file:
            pickle.dump(data, file)
    else:
        with open(save_path, "rb") as file:
            data = pickle.load(file)
    return [data, None]

def uk(root=None, save=False):
    save_path = root + "/uk-2006-08.pkl"
    if save == True:
        file = root + "/uk-2006-08.graph"
        nodes_list = []
        with open(file, "rb") as file:
            for line in file:
                nodes = int(line.strip().split()[0]) - 1
                nodes_list.append(nodes)
        if sorted(nodes_list) == list(range(len(nodes_list))):
            x = torch.zeros((len(nodes_list),1))
        else:
            raise ValueError('Wrong Nodes List!')
        file_edge = root + "/soc-pokec-relationships.txt.gz"
        edges_list = [[], []]
        with gzip.open(file_edge, "rt") as file:
            for line in file:
                u, v = line.strip().split()
                edges_list[0].append(int(u) - 1)
                edges_list[1].append(int(v) - 1)
        edge_index = torch.tensor(edges_list)
        data = Data(x=x, edge_index=edge_index)

        with open(save_path, "wb") as file:
            pickle.dump(data, file)
    else:
        with open(save_path, "rb") as file:
            data = pickle.load(file)
    return [data, None]

# def Webbase(root=None, save=False, num_parts=4):
#     # save_path = root + "/webbase.pkl"
#     if save == True:
#         file_name = root + "/webbase-2001.mtx"
#         matrix = mmread(file_name)
#         num_e = matrix.nnz
#         part_size = num_e // num_parts
#         for i in range(num_parts):
#             start = i * part_size
#             end = (i + 1) * part_size if i < num_parts - 1 else num_e
#             row = torch.tensor(matrix.row[start:end], dtype=torch.long)
#             col = torch.tensor(matrix.col[start:end], dtype=torch.long)
#             edge_index = torch.stack([row, col], dim=0)
#             data = Data(edge_index=edge_index)
#             with open(root + f"/webbase_{i}.pkl", "wb") as file:
#                 pickle.dump(data, file)

#     else:
#         edge_index_list = []
#         for i in range(num_parts):
#             with open(root + f"/webbase_{i}.pkl", "rb") as file:
#                 data_part = pickle.load(file)
#                 edge_index_list.append(data_part.edge_index)
#         edge_index = torch.cat(edge_index_list, dim=1)
#         edge_index, _ = remove_self_loops(edge_index)
#         data = Data(edge_index=edge_index, num_nodes=edge_index.max().item()+1)
#     return [data, None]
