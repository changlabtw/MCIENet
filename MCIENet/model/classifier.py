import torch
import torch.nn as nn

from .utils import get_activite_func
from collections import OrderedDict

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_layer, 
                 activite_func, slope=0.01, dropout=0.5, bn=False, bn_eps: float = 0.00001, bn_momentum: float = 0.1):
        super().__init__()

        hidden_layer_ls = []
        for idx, _ in enumerate(range(hidden_layer)):
            dim = input_dim if idx == 0 else hidden_size

            layers = [
                (f'hidden_{idx}', nn.Linear(dim, hidden_size)),
                (f'bn_{idx}', nn.BatchNorm1d(int(hidden_size), eps=bn_eps, momentum=bn_momentum)) if bn else None,
                (f'activite_func_{idx}', get_activite_func(activite_func, slope=slope)),
                (f'dropout_{idx}', nn.Dropout(dropout)) if dropout!=0 else None
                ]
            layers = [i for i in layers if i != None]

            hidden_layer_ls.extend(layers)

        self.hidden_net = nn.Sequential(OrderedDict(hidden_layer_ls))

        self.out = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.hidden_net(x) # [batch_size, hidden_layer] -> ... -> [batch_size, hidden_layer]

        x = self.out(x) # [batch_size, output_dim]

        return x

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_layer, 
                 activite_func, nhead=8, num_encoder_layers=3, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", slope=0.01, bn=False, 
                 bn_eps: float = 0.00001, bn_momentum: float = 0.1):
        """
        Transformer-based classifier that processes features from the data extractor.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output
            hidden_size: Hidden size of the transformer model
            hidden_layer: Number of hidden layers (not used in transformer, kept for interface compatibility)
            activite_func: Activation function name
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of the feedforward network in transformer
            dropout: Dropout probability
            activation: Activation function in transformer
            slope: Slope for leaky relu (if used)
            bn: Whether to use batch normalization
            bn_eps: BatchNorm epsilon
            bn_momentum: BatchNorm momentum
        """
        super().__init__()
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Batch normalization if enabled
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(input_dim, eps=bn_eps, momentum=bn_momentum)
        
        # Activation function
        self.activite_func = get_activite_func(activite_func, slope=slope)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.out = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass of the transformer classifier.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Add sequence dimension if not present (for compatibility with single-sequence inputs)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Transformer expects [seq_len, batch_size, input_dim]
        x = x.transpose(0, 1)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Take the output corresponding to the first token (CLS token equivalent)
        x = x[0]  # [batch_size, input_dim]
        
        # Apply batch norm if enabled
        if self.bn is not None:
            x = self.bn(x)
        
        # Apply activation
        x = self.activite_func(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final output layer
        x = self.out(x)
        
        return x

# =================================