"""Dataset classes for handling sparse matrices and variable data in PyTorch."""

import numpy as np
import torch
from scipy.sparse import csr_matrix


class TorchSparseDataset:
    """A dataset class for handling probability matrices in PyTorch.

    Stores sparse COO data as sorted linear indices and values on the
    target device.  Look-ups use ``torch.searchsorted`` so everything
    stays on-device (CPU, CUDA, or MPS) with O(nnz) memory and
    O(batch * log nnz) query time.

    Parameters
    ----------
    P_sym : csr_matrix
        Symmetric probability matrix
    device : str, optional
        Device for the backing tensors, by default 'cpu'

    """

    def __init__(self, P_sym: csr_matrix, device: str = "cpu") -> None:
        """Initialize the dataset from a CSR probability matrix."""
        coo = P_sym.tocoo()
        n = P_sym.shape[0]

        rows = torch.tensor(coo.row, dtype=torch.long)
        cols = torch.tensor(coo.col, dtype=torch.long)
        keys = rows * n + cols

        # Sort by linear index so searchsorted works.
        order = torch.argsort(keys)
        self._keys = keys[order].to(device)
        self._values = torch.tensor(coo.data, dtype=torch.float32)[order].to(device)
        self._n = n
        self._nnz = P_sym.nnz
        self.device = device

    def __getitem__(self, idx: tuple[int, int] | list[tuple[int, int]]) -> torch.Tensor:
        """Look up probability values for one or more (i, j) pairs.

        Parameters
        ----------
        idx : Union[Tuple[int, int], List[Tuple[int, int]]]
            Either a single (i, j) tuple or a list of such tuples.

        Returns
        -------
        torch.Tensor
            Requested values (0 where no edge exists).

        """
        if isinstance(idx, tuple) and isinstance(idx[0], int):
            query = torch.tensor([idx[0] * self._n + idx[1]], dtype=torch.long, device=self.device)
        else:
            rows, cols = zip(*idx, strict=False)
            rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
            cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
            query = rows_t * self._n + cols_t

        if self._nnz == 0:
            result = torch.zeros(query.shape, dtype=torch.float32, device=self.device)
        else:
            pos = torch.searchsorted(self._keys, query).clamp(max=self._nnz - 1)
            found = self._keys[pos] == query
            result = torch.where(found, self._values[pos], torch.zeros_like(self._values[pos]))

        if isinstance(idx, tuple) and isinstance(idx[0], int):
            return result.squeeze()
        return result

    def __len__(self) -> int:
        """Return the number of non-zero elements."""
        return self._nnz

    def to(self, device: str) -> "TorchSparseDataset":
        """Move the backing storage to *device*.

        Parameters
        ----------
        device : str
            Target device ('cpu', 'cuda:X', or 'mps')

        Returns
        -------
        TorchSparseDataset
            Self for method chaining

        """
        self._keys = self._keys.to(device)
        self._values = self._values.to(device)
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
        """Initialize the variable dataset from a numpy array."""
        self.X = torch.as_tensor(X, dtype=torch.float32)
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
        ValueError
            If indexes_map is not initialized

        """
        if self.indexes_map is None:
            msg = "Indexes map not initialized"
            raise ValueError(msg)
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
