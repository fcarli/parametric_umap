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

class ParametricUMAP:
    def __init__(
        self,
        n_components=2,
        hidden_dim=1024,
        n_layers=3,
        n_neighbors=15,
        a=0.1,
        b=1.0,
        correlation_weight=0.1,
        learning_rate=1e-4,
        n_epochs=10,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_batchnorm=False,
        use_dropout=False
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
        self.correlation_weight = correlation_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        
        self.model = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False
        
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
        
    def fit(self, X, y=None,
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
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        losses = []

        loader = ed.get_loader(batch_size=self.batch_size, 
                               sample_first=True,
                               random_state=random_state,
                               n_processes=n_processes,
                               verbose=verbose)
        
        if verbose:
            print('Training...')
        
        pbar = tqdm(range(self.n_epochs), desc='Epochs', position=0)
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
                loss = umap_loss + self.correlation_weight * corr_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            if resample_negatives:
                loader = ed.get_loader(batch_size=self.batch_size, sample_first=True)
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            if verbose:
                print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}')
            
        self.is_fitted = True
        return self
    
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
    
    def fit_transform(self, X,verbose=True,low_memory=False):
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
        self.fit(X,verbose=verbose,low_memory=low_memory)
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