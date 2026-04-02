import os
import torch
from torch_geometric.data import Data



class GraphPartitioner:
    """Partition a large graph into target-node-owned edge blocks.

    ``FS_GPlib`` does not provide a full distributed simulation API.
    Instead, it provides a lightweight partitioning utility that splits a
    graph into several subgraphs suitable for user-managed distributed or
    out-of-core propagation loops.

    The partitioning strategy is **target-node based**: each partition owns a
    subset of destination nodes, and keeps all edges whose destination lies in
    that subset.  This design matches the message-passing update pattern used
    by the propagation models, where target nodes aggregate messages from their
    incoming neighbors.

    Each generated subgraph preserves the original global node indexing and
    stores only a filtered ``edge_index`` / ``edge_attr`` pair.  Therefore,
    the calling code is responsible for synchronising node states across
    partitions after each propagation step when running a distributed
    simulation.

    :param data: Input PyG graph. ``edge_index`` is required; ``edge_attr`` is
        optional.
    :type data: torch_geometric.data.Data
    :param n_parts: Number of partitions. Must be greater than 1.
    :type n_parts: int
    :param root: Optional directory used to persist partitions to disk.  When
        provided, each partition is saved to ``root/part_i/``.
    :type root: str | None
    """
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
        """Generate graph partitions and optionally save them to disk.

        The method proceeds in two stages:

        1. Partition target nodes into ``n_parts`` groups according to the
           number of incoming edges, using a Longest Processing Time (LPT)
           heuristic on destination-node workloads.
        2. Build one subgraph per group by keeping only edges whose
           destination node belongs to that group.

        Results are stored in ``self.sub_nodes`` and ``self.sub_data`` when
        ``root`` is ``None``.  If ``root`` is provided, each partition is
        saved to:

        - ``root/part_i/sub_data.pt``: the filtered PyG ``Data`` object
        - ``root/part_i/sub_nodes.pt``: the owned destination nodes
        """
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
        """Split destination nodes into approximately balanced partitions.

        The implementation uses a Longest Processing Time (LPT) heuristic:
        it counts the number of edges ending at each target node, sorts nodes
        by that workload in descending order, and then divides the ordered
        list into ``n_parts`` bins according to cumulative edge volume.

        :param indices: Edge index tensor of shape ``(2, E)``.
        :type indices: torch.LongTensor
        :param num_cols: Number of destination columns, typically equal to the
            total node count.
        :type num_cols: int
        :param n_parts: Number of partitions.
        :type n_parts: int
        :return: A list of tensors; each tensor contains the target nodes
            assigned to one partition.
        :rtype: list[torch.Tensor]
        """
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
        """Keep only edges whose destination nodes belong to ``cols``.

        :param cols: Destination nodes owned by a partition.
        :type cols: list | torch.Tensor
        :return: Filtered ``edge_index`` and aligned ``edge_attr``.
        :rtype: tuple[torch.Tensor, torch.Tensor | None]
        """
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
    """Load one previously saved graph partition from disk.

    :param root: Root directory passed to :class:`GraphPartitioner`.
    :type root: str
    :param partition_idx: Partition id to load.
    :type partition_idx: int
    :return: A pair ``(sub_data, sub_targets)`` where ``sub_data`` is the
        filtered PyG ``Data`` object and ``sub_targets`` is the tensor of
        destination nodes owned by that partition.
    :rtype: tuple[torch_geometric.data.Data, torch.Tensor]
    """
    path = os.path.join(root, f'part_{partition_idx}')
    sub_data = torch.load(f'{path}/sub_data.pt', weights_only=False)
    sub_targets = torch.load(f'{path}/sub_nodes.pt')
    return sub_data, sub_targets
