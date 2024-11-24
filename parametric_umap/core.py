from .models.mlp import MLP
from .datasets.edge_dataset import EdgeDataset
from .datasets.covariates_datasets import VariableDataset, TorchSparseDataset
from .utils.losses import compute_correlation_loss
from .utils.graph import compute_all_p_umap

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm.auto import tqdm
import faiss
from copy import deepcopy
from torch.autograd import Variable


class ParametricUMAP:
    def __init__(
        self,
        n_components=2,
        hidden_dim=1024,
        n_layers=3,
        n_neighbors=15,
        a=0.1,
        b=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_batchnorm=False,
        use_dropout=False,
        future_adaptation=False
    ):
        """
        Initialize ParametricUMAP.
        
        Parameters:
        -----------
        n_components : int
            Number of dimensions in the output embedding
        hidden_dim : int
            Dimension of hidden layers in the MLP
        n_layers : int
            Number of hidden layers in the MLP
        a, b : float
            UMAP parameters for the optimization
        correlation_weight : float
            Weight of the correlation loss term
        learning_rate : float
            Learning rate for the optimizer
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        device : str
            Device to use for computations ('cpu' or 'cuda')
        """
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.a = a
        self.b = b
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
        self.model = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False
        self.future_adaptation = future_adaptation
        
    def _init_model(self, input_dim):
        """Initialize the MLP model"""
        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_components,
            num_layers=self.n_layers,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout
        ).to(self.device)

    def _train_loop(self, loader, dataset, target_dataset, optimizer,
                    n_epochs=10, batch_size=32,
                    correlation_weight=0.1,
                    flag='fit',
                    ewc_lambda=0.1,
                    random_state=0, n_processes=6, 
                    verbose=True, low_memory=False,
                    resample_negatives=False):

        self.model.train()
        losses = []
        
        if verbose:
            print('Training...' if flag == 'fit' else 'Adapting...')
        
        pbar = tqdm(range(n_epochs), desc='Epochs', position=0)
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0
            
            for edge_batch in tqdm(loader, desc=f'Epoch {epoch+1}', position=1, leave=False):
                optimizer.zero_grad()
                
                # Get src and dst indexes from edge_batch
                src_indexes = [i for i,j in edge_batch]
                dst_indexes = [j for i,j in edge_batch]
                
                # Get values from dataset
                src_values = dataset[src_indexes]
                dst_values = dataset[dst_indexes]
                #print(edge_batch)
                #print(target_dataset)
                #print(target_dataset[edge_batch])
                #break
                targets = target_dataset[edge_batch]

                # If low memory, the dataset is not on GPU, so we need to move the values to GPU
                if low_memory:
                    src_values = src_values.to(self.device)
                    dst_values = dst_values.to(self.device)
                    targets = targets.to(self.device)
                
                # Get embeddings from model
                src_embeddings = self.model(src_values)
                dst_embeddings = self.model(dst_values)
                
                # Compute distances
                Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1)
                X_distances = torch.norm(src_values - dst_values, dim=1)
                
                # Compute losses
                qs = torch.pow(1 + self.a * torch.norm(src_embeddings - dst_embeddings, dim=1, p=2*self.b), -1)
                umap_loss = self.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(X_distances, Z_distances)
                loss = umap_loss + correlation_weight * corr_loss

                # Add EWC loss if in adaptation mode
                if flag == 'adapt' and hasattr(self, 'fisher_info'):

                    params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
                    p_old = {}
                    
                    for n, p in deepcopy(params).items():
                        p_old[n] = Variable(p.data)
                    
                    fisher_matrix = self._compute_fisher_information(loader, dataset, target_dataset,sub_ratio=0.1,correlation_weight=correlation_weight)
                    ewc_loss = self._get_ewc_loss(self.model, fisher_matrix, p_old)
                    print(ewc_loss,loss)
                    loss += ewc_loss * ewc_lambda
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            if resample_negatives:
                loader = ed.get_loader(batch_size=batch_size, sample_first=True)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            if verbose:
                print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}')
        
    def fit(self, X, y=None,
            n_epochs=10, batch_size=32,
            learning_rate=1e-4, correlation_weight=0.1,
            resample_negatives=False,
            n_processes=6,
            low_memory=False,
            random_state=0,
            verbose=True):
        """
        Fit the model using X as training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
        
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X).astype(np.float32)

        if self.future_adaptation:
            self._select_anchors(X)
        
        # Initialize model if not already done
        if self.model is None:
            self._init_model(X.shape[1])
            
        # Create datasets
        dataset = VariableDataset(X).to(self.device)
        P_sym = compute_all_p_umap(X, k=self.n_neighbors)
        ed = EdgeDataset(P_sym)
        
        if low_memory:
            target_dataset = TorchSparseDataset(P_sym)
        else:
            target_dataset = TorchSparseDataset(P_sym).to(self.device) #if the dataset is not too big, it's better to keep it on GPU for faster computation
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        loader = ed.get_loader(batch_size=batch_size, 
                               sample_first=True,
                               random_state=random_state,
                               n_processes=n_processes,
                               verbose=verbose)
        
        # Training loop
        self._train_loop(loader, dataset, target_dataset, optimizer, 
                         n_epochs=n_epochs, batch_size=batch_size,
                         correlation_weight=correlation_weight,
                         random_state=random_state, n_processes=n_processes, 
                         verbose=verbose, low_memory=low_memory,
                         resample_negatives=resample_negatives)
        
        if self.future_adaptation:
            print('Computing Fisher Information...')
            # Compute Fisher Information for EWC using only anchor points
            self.fisher_info = self._compute_fisher_information(
                loader, dataset, target_dataset,sub_ratio=0.1
            )
            
        self.is_fitted = True

    def adapt(self, X_new, n_processes=6, verbose=True,
              n_epochs=1, batch_size=32,
              learning_rate=1e-4, correlation_weight=0.1,
              random_state=0):
        

        X_new = np.asarray(X_new).astype(np.float32)
    
        # Combine new data with anchors
        X_combined = np.vstack([self.anchors, X_new])

        # Create mask for anchors (True for anchors, False for new points)
        anchors_mask = np.zeros(len(X_combined), dtype=bool)
        anchors_mask[:len(self.anchors)] = True
        
        # Create datasets
        dataset = VariableDataset(X_combined).to(self.device)
        
        # Compute UMAP probabilities using the combined data
        P_sym = compute_all_p_umap(X_combined, k=self.n_neighbors)
        ed = EdgeDataset(P_sym, anchors_mask=anchors_mask)
        target_dataset = TorchSparseDataset(P_sym).to(self.device)
    
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        loader = ed.get_loader(batch_size=batch_size, 
                               sample_first=True,
                               random_state=random_state,
                               n_processes=n_processes,
                               verbose=verbose)

        self._train_loop(loader, dataset, target_dataset, optimizer,
                         n_epochs=n_epochs, batch_size=batch_size,
                         correlation_weight=correlation_weight,
                         random_state=random_state, n_processes=n_processes, 
                         verbose=verbose,
                         flag='adapt')

    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to transform
            
        Returns:
        --------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")
            
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            X_reduced = self.model(X)
            
        return X_reduced.cpu().numpy()
    
    def fit_transform(self, X,verbose=True,low_memory=False,
                      n_epochs=10, batch_size=32,
                      learning_rate=1e-4, correlation_weight=0.1,
                      n_processes=6, random_state=0,
                      resample_negatives=False):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_new : array-like of shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X,verbose=verbose,low_memory=low_memory,
                n_epochs=n_epochs, batch_size=batch_size,
                learning_rate=learning_rate, correlation_weight=correlation_weight,
                n_processes=n_processes, random_state=random_state,
                resample_negatives=resample_negatives)
        return self.transform(X)
    
    def save(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'n_components': self.n_components,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'a': self.a,
            'b': self.b,
            'correlation_weight': self.correlation_weight,
            'use_batchnorm': self.use_batchnorm,
            'use_dropout': self.use_dropout
        }
        
        torch.save(save_dict, path)

    def _select_anchors(self, X, n_clusters=20, points_ratio=0.03):
        """
        Select anchor points using clustering for efficient adaptation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to select anchors from
        n_clusters : int
            Number of clusters to create
        samples_per_cluster : int
            Number of samples to select from each cluster
        """
        X = np.asarray(X).astype(np.float32)
        n_samples, n_features = X.shape

        # Initialize FAISS clustering
        n_clusters = min(n_clusters, n_samples // 2)  # Ensure we don't exceed dataset size
        samples_per_cluster = int(n_samples * points_ratio // n_clusters)
        kmeans = faiss.Kmeans(d=n_features, k=n_clusters)#, gpu=torch.cuda.is_available())
        kmeans.train(X)
        
        # Get cluster assignments
        _, labels = kmeans.index.search(X, 1)
        labels = labels.ravel()
        
        # Select samples from each cluster
        anchor_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Select min(samples_per_cluster, cluster_size) samples
                selected = np.random.choice(
                    cluster_indices,
                    size=min(samples_per_cluster, len(cluster_indices)),
                    replace=False
                )
                anchor_indices.extend(selected)
        
        # Store anchors and their embeddings
        self.anchors = X[anchor_indices]
        #with torch.no_grad():
        #    self.anchor_embeddings = self.model(
        #        torch.tensor(self.anchors, dtype=torch.float32).to(self.device)
        #    )
        
    def _compute_fisher_information(self,loader,dataset,target_dataset,
                                    sub_ratio=0.1,correlation_weight=0.1):
        """
        Compute Fisher Information Matrix for EWC using only anchor points
        """
        
        fisher = {}
        for n, p in self.model.named_parameters():
            fisher[n] = Variable(torch.zeros_like(p.data))

        self.model.eval()

        loader_len = len(loader)
        n_batches = int(loader_len * sub_ratio)

        n_pairs = 0
        batch_index = 0
        
        for edge_batch in tqdm(loader, desc='Computing Fisher Information', position=0, leave=False):
            
            self.model.zero_grad()

            # Get src and dst indexes from edge_batch
            src_indexes = [i for i,j in edge_batch]
            dst_indexes = [j for i,j in edge_batch]
                    
            # Get values from dataset
            src_values = dataset[src_indexes]
            dst_values = dataset[dst_indexes]
            targets = target_dataset[edge_batch]
                    
            # Forward pass
            src_embeddings = self.model(src_values)
            dst_embeddings = self.model(dst_values)

            # Compute distances
            Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1)
            X_distances = torch.norm(src_values - dst_values, dim=1)
                    
            # Compute losses
            qs = torch.pow(1 + self.a * torch.norm(src_embeddings - dst_embeddings, dim=1, p=2*self.b), -1)
            umap_loss = self.loss_fn(qs, targets)
            corr_loss = compute_correlation_loss(X_distances, Z_distances)
            loss = umap_loss + correlation_weight * corr_loss

            loss.backward()
                    
            # Accumulate Fisher Information
            for n, p in self.model.named_parameters():
                fisher[n].data += p.grad.data ** 2 / len(loader)

            # Count pairs
            #n_pairs += len(src_indexes)

            batch_index += 1
            if batch_index >= n_batches:
                break
        
        # Normalize by number of pairs
        fisher = {n: p for n, p in fisher.items()}
        return fisher
    
    def _get_ewc_loss(self, model, fisher, p_old):
        loss = 0
        for n, p in model.named_parameters():
            _loss = fisher[n] * (p - p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    
    @classmethod
    def load(cls, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load a saved model.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        device : str
            Device to load the model to
            
        Returns:
        --------
        model : ParametricUMAP
            Loaded model
        """
        save_dict = torch.load(path, map_location=device)
        
        # Create instance with saved parameters
        instance = cls(
            n_components=save_dict['n_components'],
            hidden_dim=save_dict['hidden_dim'],
            n_layers=save_dict['n_layers'],
            a=save_dict['a'],
            b=save_dict['b'],
            correlation_weight=save_dict['correlation_weight'],
            device=device,
            use_batchnorm=save_dict['use_batchnorm'],
            use_dropout=save_dict['use_dropout']
        )
        
        # Initialize model architecture
        instance._init_model(input_dim=save_dict['model_state_dict']['model.0.weight'].shape[1])
        
        # Load state dict
        instance.model.load_state_dict(save_dict['model_state_dict'])
        instance.is_fitted = True
        
        return instance