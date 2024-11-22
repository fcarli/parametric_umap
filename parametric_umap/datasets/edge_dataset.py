import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm


class EdgeBatchIterator:
    def __init__(self, edges, batch_size, shuffle=False, stratify=False):
        self.edges = edges
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current = 0  # Initialize current counter in __init__
        self.current_edges = self.edges # Initialize current_edges in __init__
        self.stratify = stratify
        
    def __iter__(self):
        if self.shuffle:
            # Create a copy to avoid modifying original edges
            self.current_edges = self.edges.copy()
            np.random.shuffle(self.current_edges)
        else:
            self.current_edges = self.edges

        #TODO: add stratify with respect to fake and true edges
        #if self.stratify:

        self.current = 0  # Reset counter when iterator starts
        
        return self
        
    def __next__(self):
        if self.current_edges is None:
            raise StopIteration
            
        if self.current >= len(self.current_edges):
            raise StopIteration
            
        edge_batch = self.current_edges[self.current:self.current + self.batch_size]
        self.current += self.batch_size
        return edge_batch
    
    def __len__(self):
        return (len(self.current_edges) + self.batch_size - 1) // self.batch_size


class EdgeDataset:
    def __init__(self, P_sym):
        self.adj_sets = self._get_adjacency_sets(P_sym)
        
        P_sym_dok = P_sym.todok()
        self.pos_edges = list(P_sym_dok.keys())
        
        self.neg_edges = None
        self.all_edges = None

    def _shuffle_edges(self, random_state=0):
        """Shuffle the edges using the given random state"""
        # Get permutation indices
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(len(self.all_edges))
        
        # Apply permutation to edges
        self.all_edges = [self.all_edges[i] for i in perm]
        #self.all_edges_prob = [self.all_edges_prob[i] for i in perm]
        
    def sample_and_shuffle(self, random_state=0,n_processes=6,verbose=True):
        self.sample_negative_edges(random_state=random_state,n_processes=n_processes,verbose=verbose)
        self.all_edges = self.pos_edges + self.neg_edges
        #self.all_edges_prob = self.pos_edges_prob.extend(self.neg_edges_prob)
        
        self._shuffle_edges(random_state=random_state)

    def get_loader(self, batch_size, sample_first=False, random_state=0,n_processes=6,verbose=True):
        """
        Returns an iterator that yields batches of edges and their probabilities.
        
        Parameters:
        - batch_size: int, size of each batch
        - sample_first: bool, whether to call sample_and_shuffle() before creating loader
        - random_state: int or None, random seed for reproducibility
        - n_processes: int, number of processes to use for sampling negative edges
        
        Returns:
        - Iterator yielding tuples of (edge_batch, prob_batch)
        """
        if sample_first:
            self.sample_and_shuffle(random_state=random_state,n_processes=n_processes,verbose=verbose)
            
        if self.all_edges is None:
            raise ValueError("Must call sample_and_shuffle() before getting loader")
            
        # Return a custom iterator class that can be reused
        return EdgeBatchIterator(self.all_edges, batch_size)
    
    def sample_negative_edges(self, random_state=0,n_processes=6,verbose=True):
        self.neg_edges = self._sample_negative_edges([src for src, _ in self.pos_edges], random_state=random_state,n_processes=n_processes,verbose=verbose)

    def _sample_negative_edges(self, node_list, k=5, random_state=0,n_processes=6,verbose=True):
        #count available cores
        if n_processes == -1:
            n_processes = os.cpu_count()
        else:
            n_processes = min(n_processes, os.cpu_count())

        # Create a base RNG to generate seeds for each process
        base_rng = np.random.RandomState(random_state)
        process_seeds = base_rng.randint(0, np.iinfo(np.int32).max, size=n_processes)

        # Split node_list into chunks
        node_chunks = np.array_split(node_list, n_processes)

        if verbose:
            print('Sampling negative edges...')

        # Run parallel processing with unique seeds
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            for i, (chunk, seed) in enumerate(zip(node_chunks, process_seeds)):
                futures.append(executor.submit(
                    self._sample_negative_edges_chunk, 
                    chunk, 
                    k=k, 
                    random_state=seed
                ))
            
            # Use position=0 to ensure proper display with nested progress bars
            neg_edges = []
            for future in tqdm(futures, total=len(futures), desc="Completed processes", position=0, leave=True):
                neg_edges.extend(future.result())

        return neg_edges

    def _sample_negative_edges_chunk(self, node_list, k=5, random_state=None,process_id=0):
        """
        Sample k negative edges for each node in the node list.
        A negative edge is an edge between nodes that are not connected in adj_sets.
        
        Parameters:
        - node_list: list of integers, nodes to sample negative edges for
        - adj_sets: dict of sets, adjacency sets for each node
        - k: int, number of negative edges to sample per node
        - random_state: int or None, random seed for reproducibility
    
        Returns:
        - neg_edges: list of tuples (i,j) representing negative edges
        """
        # Remove process_id parameter since we now use unique seeds
        rng = np.random.RandomState(random_state)
        n_nodes = len(self.adj_sets)
        neg_edges = []
        
        # Use position=1 for nested progress bar
        for node in tqdm(node_list, desc="Processing nodes", position=1, leave=False):
            # Get the set of nodes that are already connected
            connected = self.adj_sets[node]
            
            # Create array of all possible target nodes
            all_nodes = np.arange(n_nodes)
            
            # Create mask for nodes that are not connected
            mask = ~np.isin(all_nodes, list(connected))
            mask[node] = False  # Exclude self-loops
            
            # Get array of possible negative edge targets
            candidates = all_nodes[mask]
            
            # Sample k targets without replacement
            if len(candidates) >= k:
                targets = rng.choice(candidates, size=k, replace=False)
            else:
                # If fewer candidates than k, take all available
                targets = candidates
                
            # Add the negative edges as tuples
            for target in targets:
                neg_edges.append((node, target))
                #print(f"({node}, {target})")
            
        #remove duplicates
        neg_edges = list(set(neg_edges))
        
        return neg_edges
    
    def _get_adjacency_sets(self, P_sym):
        """
        Get the adjacency set for each node in the graph represented by P_sym.
        
        Parameters:
        - P_sym: scipy.sparse.csr_matrix, symmetric probability matrix
        
        Returns:
        - adj_sets: list of sets, where adj_sets[i] contains the indices of nodes 
                connected to node i with non-zero probability
        """
        n_samples = P_sym.shape[0]
        adj_sets = []
        
        # Convert to COO format for efficient iteration over non-zero elements
        P_coo = P_sym.tocoo()
        
        # Initialize empty sets for each node
        for _ in range(n_samples):
            adj_sets.append(set())
            
        # Iterate through non-zero elements and add to adjacency sets
        for i, j, val in zip(P_coo.row, P_coo.col, P_coo.data):
            if val > 0:
                adj_sets[i].add(j)
                
        return {i: set(adj_sets[i]) for i in range(n_samples)}
    
