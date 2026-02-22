"""Edge dataset classes for graph-based training of parametric UMAP."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


class EdgeBatchIterator:
    """Iterator class for batching edges during training.

    Parameters
    ----------
    edges : List[Tuple[int, int]]
        List of edges to iterate over
    batch_size : int
        Size of each batch
    shuffle : bool, optional
        Whether to shuffle edges before iteration, by default False
    stratify : bool, optional
        Whether to stratify batches by edge type, by default False

    """

    def __init__(
        self,
        edges: list[tuple[int, int]],
        batch_size: int,
        shuffle: bool = False,
        stratify: bool = False,
    ) -> None:
        """Initialize the edge batch iterator."""
        self.edges = edges
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current: int = 0
        self.current_edges: list[tuple[int, int]] = self.edges
        self.stratify = stratify

    def __iter__(self) -> "EdgeBatchIterator":
        """Return the iterator, optionally shuffling edges."""
        if self.shuffle:
            # Create a copy to avoid modifying original edges
            self.current_edges = self.edges.copy()
            np.random.shuffle(self.current_edges)
        else:
            self.current_edges = self.edges

        # TODO: add stratify with respect to fake and true edges
        # if self.stratify:

        self.current = 0  # Reset counter when iterator starts

        return self

    def __next__(self) -> list[tuple[int, int]]:
        """Return the next batch of edges."""
        if self.current_edges is None:
            raise StopIteration

        if self.current >= len(self.current_edges):
            raise StopIteration

        edge_batch = self.current_edges[self.current : self.current + self.batch_size]
        self.current += self.batch_size
        return edge_batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.current_edges) + self.batch_size - 1) // self.batch_size


class EdgeDataset:
    """Dataset class for handling graph edges, including positive and negative edge sampling.

    Parameters
    ----------
    P_sym : csr_matrix
        Symmetric probability matrix representing the graph

    """

    def __init__(self, P_sym: csr_matrix) -> None:
        """Initialize the edge dataset from a symmetric probability matrix."""
        self.adj_sets: dict[int, set[int]] = self._get_adjacency_sets(P_sym)

        P_sym_dok = P_sym.todok()
        self.pos_edges: list[tuple[int, int]] = list(P_sym_dok.keys())

        self.neg_edges: list[tuple[int, int]] | None = None
        self.all_edges: list[tuple[int, int]] | None = None

    def _shuffle_edges(self, random_state: int = 0) -> None:
        """Shuffle all edges using the given random state.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0

        """
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(self.all_edges))

        # Apply permutation to edges
        self.all_edges = [self.all_edges[i] for i in perm]

    def sample_and_shuffle(self, random_state: int = 0, n_processes: int = 6, verbose: bool = True) -> None:
        """Sample negative edges and shuffle all edges.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True

        """
        self.sample_negative_edges(random_state=random_state, n_processes=n_processes, verbose=verbose)
        self.all_edges = self.pos_edges + self.neg_edges

        self._shuffle_edges(random_state=random_state)

    def get_loader(
        self,
        batch_size: int,
        sample_first: bool = False,
        random_state: int = 0,
        n_processes: int = 6,
        verbose: bool = True,
    ) -> EdgeBatchIterator:
        """Return an iterator that yields batches of edges and their probabilities.

        Parameters
        ----------
        batch_size : int
            Size of each batch
        sample_first : bool, optional
            Whether to sample and shuffle edges before creating loader, by default False
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True

        Returns
        -------
        EdgeBatchIterator
            Iterator yielding batches of edges

        """
        if sample_first:
            self.sample_and_shuffle(random_state=random_state, n_processes=n_processes, verbose=verbose)

        if self.all_edges is None:
            raise ValueError("Must call sample_and_shuffle() before getting loader")

        # Return a custom iterator class that can be reused
        return EdgeBatchIterator(self.all_edges, batch_size)

    def sample_negative_edges(self, random_state: int = 0, n_processes: int = 6, verbose: bool = True) -> None:
        """Sample negative edges for the graph.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True

        """
        self.neg_edges = self._sample_negative_edges(
            [src for src, _ in self.pos_edges],
            random_state=random_state,
            n_processes=n_processes,
            verbose=verbose,
        )

    def _sample_negative_edges(
        self,
        node_list: list[int],
        k: int = 5,
        random_state: int = 0,
        n_processes: int = 6,
        verbose: bool = True,
    ) -> list[tuple[int, int]]:
        """Sample k negative edges for each node in parallel.

        Parameters
        ----------
        node_list : List[int]
            List of nodes to sample negative edges for
        k : int, optional
            Number of negative edges per node, by default 5
        random_state : int, optional
            Random seed for reproducibility, by default 0
        n_processes : int, optional
            Number of processes for parallel sampling, by default 6
        verbose : bool, optional
            Whether to show progress bars, by default True

        Returns
        -------
        List[Tuple[int, int]]
            List of sampled negative edges

        """
        n_processes = os.cpu_count() if n_processes == -1 else min(n_processes, os.cpu_count())

        if verbose:
            print("Sampling negative edges...")

        # Fast path: skip multiprocessing overhead when single-threaded
        if n_processes <= 1:
            return self._sample_negative_edges_chunk(node_list, k=k, random_state=random_state)

        # Create a base RNG to generate seeds for each process
        base_rng = np.random.RandomState(random_state)
        process_seeds = base_rng.randint(0, np.iinfo(np.int32).max, size=n_processes)

        # Split node_list into chunks
        node_chunks = np.array_split(node_list, n_processes)

        # Use "spawn" context to avoid deadlocks when OpenMP / PyTorch / FAISS
        # have already been loaded in the parent process (default "fork" on
        # macOS copies the OMP runtime state and can hang or segfault).
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_processes, mp_context=ctx) as executor:
            futures = []
            for chunk, seed in zip(node_chunks, process_seeds, strict=False):
                futures.append(
                    executor.submit(
                        self._sample_negative_edges_chunk,
                        chunk,
                        k=k,
                        random_state=seed,
                    ),
                )

            # Use position=0 to ensure proper display with nested progress bars
            neg_edges = []
            for future in tqdm(futures, total=len(futures), desc="Completed processes", position=0, leave=True):
                neg_edges.extend(future.result())

        return neg_edges

    def _sample_negative_edges_chunk(
        self,
        node_list: list[int],
        k: int = 5,
        random_state: int | None = None,
    ) -> list[tuple[int, int]]:
        """Sample k negative edges for each node in the node list.

        A negative edge is an edge between nodes that are not connected in adj_sets.

        Parameters
        ----------
        node_list : List[int]
            List of nodes to sample negative edges for
        k : int, optional
            Number of negative edges per node, by default 5
        random_state : Optional[int], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        List[Tuple[int, int]]
            List of sampled negative edges for the chunk

        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        # Remove process_id parameter since we now use unique seeds
        rng = np.random.RandomState(random_state)
        n_nodes = len(self.adj_sets)
        neg_edges = []

        # Use position=1 for nested progress bar
        for node in tqdm(node_list, desc="Processing nodes", position=1, leave=False):
            connected = self.adj_sets[node]
            max_candidates = n_nodes - len(connected) - 1  # exclude self and neighbors

            if max_candidates <= 0:
                continue

            n_samples = min(k, max_candidates)
            targets: set[int] = set()
            while len(targets) < n_samples:
                candidate = rng.randint(0, n_nodes)
                if candidate != node and candidate not in connected:
                    targets.add(candidate)

            neg_edges.extend((node, target) for target in targets)

        # remove duplicates
        neg_edges = list(set(neg_edges))

        return neg_edges

    def _get_adjacency_sets(self, P_sym: csr_matrix) -> dict[int, set[int]]:
        """Get the adjacency set for each node in the graph represented by P_sym.

        Parameters
        ----------
        P_sym : csr_matrix
            Symmetric probability matrix

        Returns
        -------
        Dict[int, Set[int]]
            Dictionary mapping each node to its set of neighbors

        """
        n_samples = P_sym.shape[0]
        adj_sets = []

        # Convert to COO format for efficient iteration over non-zero elements
        P_coo = P_sym.tocoo()

        # Initialize empty sets for each node
        adj_sets = [set() for _ in range(n_samples)]

        # Iterate through non-zero elements and add to adjacency sets
        for i, j, val in zip(P_coo.row, P_coo.col, P_coo.data, strict=False):
            if val > 0:
                adj_sets[i].add(j)

        return {i: set(adj_sets[i]) for i in range(n_samples)}
