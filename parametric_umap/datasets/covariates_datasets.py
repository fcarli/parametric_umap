import numpy as np
import torch
from scipy.sparse import csr_matrix


class TorchSparseDataset:
    """A dataset class for handling sparse matrices in PyTorch.

    Parameters
    ----------
    P_sym : csr_matrix
        Symmetric probability matrix to convert to torch sparse format
    device : str, optional
        Device to store the sparse tensor on ('cpu' or 'cuda:X'), by default 'cpu'

    """

    def __init__(self, P_sym: csr_matrix, device: str = "cpu") -> None:
        coo = P_sym.tocoo()
        values = torch.FloatTensor(coo.data)
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))

        # Create sparse tensor and move to device
        self.P_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=P_sym.shape,
            device=device,
        )
        self.device = device

    def __getitem__(self, idx: tuple[int, int] | list[tuple[int, int]]) -> torch.Tensor:
        """Get elements from the sparse tensor.

        Parameters
        ----------
        idx : Union[Tuple[int, int], List[Tuple[int, int]]]
            Either a single tuple of (i,j) indices or a list of such tuples

        Returns
        -------
        torch.Tensor
            Tensor containing the requested values, 0 if index not found

        """
        if isinstance(idx, tuple) and isinstance(idx[0], int):
            # Single index access
            i, j = idx
            indices = torch.tensor([[i], [j]], device=self.device)
            return self.P_sparse.index_select(0, indices[0]).index_select(1, indices[1]).to_dense().squeeze()
        # Multiple index access
        indices = torch.tensor(list(zip(*idx, strict=False)), device=self.device)
        values = self.P_sparse.index_select(0, indices[0]).index_select(1, indices[1]).to_dense().diagonal()
        return values

    def __len__(self) -> int:
        """Get the number of non-zero elements in the sparse tensor.

        Returns
        -------
        int
            Number of non-zero elements

        """
        return self.P_sparse._nnz()

    def to(self, device: str) -> "TorchSparseDataset":
        """Move the sparse tensor to the specified device.

        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda:X')

        Returns
        -------
        TorchSparseDataset
            Self for method chaining

        """
        self.P_sparse = self.P_sparse.to(device)
        self.device = device
        return self


class VariableDataset:
    """A dataset class for handling variable data in PyTorch.

    Parameters
    ----------
    X : np.ndarray
        Input data array
    indexes : Optional[List[int]], optional
        Optional list of indexes to map positions in X, by default None

    """

    def __init__(self, X: np.ndarray, indexes: list[int] | None = None) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.indexes_map: dict[int, int] | None = None

        if indexes is not None:
            self.indexes_map = {idx: i for i, idx in enumerate(indexes)}

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples

        """
        return len(self.X)

    def to(self, device: str) -> "VariableDataset":
        """Move the data tensor to the specified device.

        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda:X')

        Returns
        -------
        VariableDataset
            Self for method chaining

        """
        self.X = self.X.to(device)
        return self

    def get_index(self, idx: int) -> int:
        """Get the position in X for a given index.

        Parameters
        ----------
        idx : int
            Index to look up

        Returns
        -------
        int
            Position in X corresponding to the index

        Raises
        ------
        AssertionError
            If indexes_map is not initialized

        """
        assert self.indexes_map is not None, "Indexes map not initialized"
        return self.indexes_map[idx]

    def get_values_by_indexes(self, indexes: list[int]) -> torch.Tensor:
        """Get values from X using a list of indexes.

        Parameters
        ----------
        indexes : List[int]
            List of indexes to look up

        Returns
        -------
        torch.Tensor
            Tensor containing the values at the specified indexes

        """
        return self.X[[self.get_index(idx) for idx in indexes]]

    def __getitem__(self, idx: int | list[int]) -> torch.Tensor:
        """Get items from the dataset.

        Parameters
        ----------
        idx : Union[int, List[int]]
            Index or list of indexes to retrieve

        Returns
        -------
        torch.Tensor
            Tensor containing the requested values

        """
        return self.X[idx]
