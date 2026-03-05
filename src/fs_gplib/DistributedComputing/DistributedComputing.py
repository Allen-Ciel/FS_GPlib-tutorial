import os
import torch
from torch_geometric.data import Data



class GraphPartitioner:
    def __init__(self, data: Data, n_parts: int, root: str = None):
        self.data = data

        if self.data.x == None:
            num = data.edge_index.max().item()+1
            self.num_nodes = num
        else:
            self.num_nodes = self.data.x.shape[0]
        if self.data.edge_attr == None:
            self.have_weight = False
        else:
            self.have_weight = True

        assert n_parts > 1
        self.n_parts = n_parts

        self.root = root
        self.store_data = False if root is None else True

    def generate_partition(self):
        # 1. LPT
        self.sub_nodes = self.lpt_partition_by_columns(
            self.data.edge_index,
            self.num_nodes,
            self.n_parts
        )
        # 2. subgraph Data
        self.sub_data = {}
        for i in range(self.n_parts):
            edge_index, edge_attr = self._filter_edges(self.sub_nodes[i])
            self.sub_data[i] = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.num_nodes)


        if self.store_data:
            for i in range(self.n_parts):
                path = os.path.join(self.root, f'part_{i}')
                os.makedirs(path, exist_ok=True)
                torch.save(self.sub_data[i], f'{path}/sub_data.pt')
                torch.save(self.sub_nodes[i], f'{path}/sub_nodes.pt')
                del self.sub_data[i],
            del self.sub_nodes

    def lpt_partition_by_columns(self, indices: torch.LongTensor, num_cols: int, n_parts: int):
        col_idx = indices[1]
        col_nnz = torch.bincount(col_idx, minlength=num_cols)

        cols_sorted = torch.argsort(col_nnz, descending=True)
        weights = col_nnz[cols_sorted]
        cumsum = torch.cumsum(weights, dim=0)
        total = float(cumsum[-1])

        part_bins = []
        start_idx = 0
        for j in range(1, n_parts + 1):
            thresh = total * j / n_parts
            end_idx = int(torch.searchsorted(cumsum, torch.tensor(thresh)).item())
            part_bins.append(cols_sorted[start_idx:end_idx])
            start_idx = end_idx
        return part_bins

    def _filter_edges(self,cols: list):
        edge_index = self.data.edge_index
        device = edge_index.device
        valid = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        valid[cols] = True
        # Mask edges whose destination node is in cols
        mask = valid[edge_index[1]]

        if self.have_weight:
            return edge_index[:, mask], self.data.edge_attr[mask]
        else:
            return edge_index[:, mask], None

def load_partition(root: str, partition_idx: int):
    path = os.path.join(root, f'part_{partition_idx}')
    sub_data = torch.load(f'{path}/sub_data.pt', weights_only=False)
    sub_targets = torch.load(f'{path}/sub_nodes.pt')
    return sub_data, sub_targets

