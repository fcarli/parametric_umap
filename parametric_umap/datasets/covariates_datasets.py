
import numpy as np
import torch

class TorchSparseDataset:
    def __init__(self, P_sym, device='cpu'):
        """
        Initialize the dataset from a scipy sparse matrix.
        
        Parameters:
        - P_sym: scipy.sparse.csr_matrix, symmetric probability matrix
        - device: str, device to store the sparse tensor on ('cpu' or 'cuda:X')
        """
        # Convert scipy sparse to torch sparse
        coo = P_sym.tocoo()
        values = torch.FloatTensor(coo.data)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        
        # Create sparse tensor and move to device
        self.P_sparse = torch.sparse_coo_tensor(
            indices, values, 
            size=P_sym.shape,
            device=device
        )
        self.device = device
        
    def __getitem__(self, idx):
        """
        Get elements from the dataset.
        
        Parameters:
        - idx: can be either:
               - tuple of (i,j) indices
               - list/array of (i,j) tuples
        
        Returns:
        - torch.Tensor containing the requested values, 0 if index not found
        """
        if isinstance(idx, tuple) and isinstance(idx[0], int):
            # Single index access
            i, j = idx
            indices = torch.tensor([[i], [j]], device=self.device)
            return self.P_sparse.index_select(0, indices[0]).index_select(1, indices[1]).to_dense().squeeze()
        else:
            # Multiple index access
            # Convert idx to tensor of indices
            indices = torch.tensor(list(zip(*idx)), device=self.device)
            values = self.P_sparse.index_select(0, indices[0]).index_select(1, indices[1]).to_dense().diagonal()
            return values
            
    def __len__(self):
        return self.P_sparse._nnz()  # Number of non-zero elements

    def to(self, device):
        """
        Moves the sparse tensor to the specified device
        
        Parameters:
        - device: str, target device ('cpu' or 'cuda:X')
        
        Returns:
        - self for chaining
        """
        self.P_sparse = self.P_sparse.to(device)
        self.device = device
        return self
    

class VariableDataset:
    def __init__(self, X,indexes=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        #map indexes to positions in X
        if indexes is not None:
            self.indexes_map = {idx: i for i, idx in enumerate(indexes)}
        
    def __len__(self):
        return len(self.X)
    
    def to(self, device):
        self.X = self.X.to(device)
        return self
    
    def get_index(self, idx):
        assert self.indexes_map is not None, "Indexes map not initialized"
        return self.indexes_map[idx]
    
    def get_values_by_indexes(self, indexes):
        return self.X[[self.get_index(idx) for idx in indexes]]
    
    def __getitem__(self, idx):
        return self.X[idx]