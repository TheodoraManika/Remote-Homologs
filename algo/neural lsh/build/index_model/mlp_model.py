"""
Neural-LSH MLP Model
====================
Multi-layer perceptron for learning hash functions.
"""

import torch
import torch.nn as nn
from typing import List


class NeuralLSH_MLP(nn.Module):
    """
    MLP Classifier for Partition Learning in Neural-LSH.
    
    Given data points p_i ∈ R^d and KaHIP partition labels π(p_i) ∈ {1,...,m},
    learns a neural map using multilayer perceptron f_θ with ReLU activations:
    
        M : R^d → R^m, M(p) = softmax(f_θ(p))
    
    Training minimizes Cross-Entropy loss: L(θ) = (1/n) Σ CE(M(p_i), π(p_i))
    """
    
    def __init__(self, input_dim: int, num_layers: int = 3, 
                 nodes_per_layer: int = 512, output_dim: int = None):
        """
        Initialize the Neural-LSH MLP classifier.
        
        Architecture: 3-layer fully connected network
        - Input layer: input_dim → nodes_per_layer
        - Hidden layer: nodes_per_layer → nodes_per_layer (with ReLU)
        - Hidden layer: nodes_per_layer → nodes_per_layer (with ReLU)
        - Output layer: nodes_per_layer → output_dim (logits)
        
        Args:
            input_dim: Dimension of input data (d)
            num_layers: Number of hidden layers (default: 3 for 3-layer network)
            nodes_per_layer: Number of nodes in each hidden layer (default: 512)
            output_dim: Output dimension = number of buckets/partitions (m)
        """
        super(NeuralLSH_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.output_dim = output_dim or nodes_per_layer
        
        # Build the network: Linear → ReLU → Linear → ReLU → Linear → (Softmax in loss)
        layers = []
        current_dim = input_dim
        
        # Hidden layers: each gets input → nodes_per_layer, then ReLU activation
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, nodes_per_layer))
            layers.append(nn.ReLU())
            current_dim = nodes_per_layer
        
        # Output layer: produces logits (softmax applied during training via loss function)
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: f_θ(p) - returns logits for softmax classification.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Logits tensor (batch_size, output_dim) - to be passed to CrossEntropyLoss
        """
        return self.network(x)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution M(p) = softmax(f_θ(p)).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Probability distribution (batch_size, output_dim)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
    def get_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get discretized bin assignment: p_r = Pr[π(p) = r] (argmax).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Hash assignments/bin assignments (batch_size,) with values in {0,...,m-1}
        """
        probabilities = self.get_probabilities(x)
        hash_assignments = torch.argmax(probabilities, dim=-1)
        return hash_assignments
    
    def get_config(self) -> dict:
        """Get the model configuration."""
        return {
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'nodes_per_layer': self.nodes_per_layer,
            'output_dim': self.output_dim,
            'architecture': '3-layer MLP classifier',
            'activation': 'ReLU',
            'output_activation': 'softmax (in loss function)'
        }
